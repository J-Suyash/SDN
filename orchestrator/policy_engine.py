"""
Policy Engine for SDN Traffic Management.

Maps classification and prediction outputs to network actions and
enforces them through the QoS Enforcer module.

Features:
- ML-based traffic classification (primary)
- SNI-based traffic classification (fallback)
- Port-based heuristics (last resort)
- Priority queue assignment (P0-P3)
- Proactive rerouting for P3 (Banking) on predicted congestion
- Reactive rerouting for P0 (Bulk) on actual congestion
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from orchestrator.sni_classifier import SNIClassifier, classify_sni
from orchestrator.qos_enforcer import (
    QoSEnforcer,
    FlowMatch,
    PathChoice,
    flow_match_from_dict,
)

# Import ML classifier (with graceful fallback)
try:
    from orchestrator.ml_classifier import MLTrafficClassifier, get_classifier

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger("policy_engine")


class Action(Enum):
    """Network policy actions."""

    ALLOW = "allow"
    THROTTLE = "throttle"
    REROUTE = "reroute"
    PRIORITY_QUEUE = "priority_queue"


@dataclass
class PolicyDecision:
    """A policy decision for a flow."""

    flow_id: str
    action: Action
    priority: str
    reason: str
    parameters: Dict[str, Any]

    # Flow details for enforcement
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 6  # TCP

    def __str__(self):
        return f"Flow {self.flow_id} ({self.priority}): {self.action.value} - {self.reason}"


class PolicyEngine:
    """
    Applies network policies based on flow classification and congestion predictions.

    Priority Rules:
    - P3 (Banking): Always priority queue, PROACTIVE reroute if congestion predicted
    - P2 (Voice): Priority queue, maintain low jitter (no reroute)
    - P1 (Web): Best effort, no special handling
    - P0 (Bulk): Lowest priority, REACTIVE reroute when congestion detected

    Reroute Strategy:
    - P3: Proactive - reroute when congestion is PREDICTED (protect latency-sensitive payments)
    - P0: Reactive - reroute when congestion is DETECTED (less critical, save alternate path)

    Usage:
        engine = PolicyEngine()
        decisions = engine.apply(flows, link_predictions)
    """

    # QoS queue mapping (matches setup_qos.sh)
    PRIORITY_QUEUES = {
        "P3": 0,  # Highest priority queue (Banking)
        "P2": 1,  # Voice/Video queue
        "P1": 2,  # Web/Best effort queue
        "P0": 3,  # Bulk/Background queue (throttled)
    }

    # Congestion thresholds
    CONGESTION_THRESHOLD = 0.80  # 80% utilization = congested
    PREDICTION_THRESHOLD = 0.70  # 70% utilization = predict congestion soon

    def __init__(self, dry_run: bool = False, use_ml: bool = True):
        """
        Initialize the Policy Engine.

        Args:
            dry_run: If True, log OVS commands without executing them.
            use_ml: If True, use ML classifier when available.
        """
        self.sni_classifier = SNIClassifier()
        self.qos_enforcer = QoSEnforcer(dry_run=dry_run)
        self.dry_run = dry_run
        self.use_ml = use_ml

        # Initialize ML classifier if available
        self.ml_classifier = None
        if use_ml and ML_AVAILABLE:
            try:
                self.ml_classifier = get_classifier()
                if self.ml_classifier.is_model_loaded():
                    logger.info("ML classifier loaded successfully")
                else:
                    logger.warning("ML classifier available but model not loaded")
                    self.ml_classifier = None
            except Exception as e:
                logger.warning(f"Failed to initialize ML classifier: {e}")

        # Track active decisions by flow_id
        self.active_decisions: Dict[str, PolicyDecision] = {}

        # Track rerouted flows to avoid flip-flopping
        self.rerouted_flows: Dict[str, float] = {}  # flow_id -> reroute_time
        self.reroute_cooldown = 30.0  # seconds before allowing re-reroute

        # Statistics
        self.stats = {
            "decisions_made": 0,
            "qos_rules_applied": 0,
            "reroutes_executed": 0,
            "classification_by_ml": 0,
            "classification_by_sni": 0,
            "classification_by_port": 0,
        }

    def classify_flow(self, flow: Dict[str, Any]) -> str:
        """
        Classify a flow to a priority class.

        Classification hierarchy:
        1. ML model (if available and confident)
        2. SNI-based classification
        3. Port-based heuristics (fallback)

        Args:
            flow: Flow dictionary with traffic features

        Returns:
            Priority label: "P0", "P1", "P2", or "P3"
        """
        # Try ML classification first
        if self.ml_classifier is not None:
            result = self.ml_classifier.classify(flow)
            if result.method == "ml":
                self.stats["classification_by_ml"] += 1
                logger.debug(
                    f"ML classified flow as {result.priority} (conf={result.confidence:.2f})"
                )
                return result.priority
            elif result.method == "sni":
                self.stats["classification_by_sni"] += 1
                return result.priority
            else:
                self.stats["classification_by_port"] += 1
                return result.priority

        # Fallback: Try SNI-based classification
        sni = flow.get("sni", "")
        if sni and sni.lower() not in ("unknown", "none", ""):
            priority = self.sni_classifier.classify(sni)
            if priority != "P1":  # P1 is default, so SNI was useful
                self.stats["classification_by_sni"] += 1
                return priority

        # Fallback to port-based heuristics
        dst_port = flow.get("dst_port", 0)
        src_port = flow.get("src_port", 0)

        priority = self._classify_by_port(dst_port, src_port)
        self.stats["classification_by_port"] += 1
        return priority

    def _classify_by_port(self, dst_port: int, src_port: int) -> str:
        """Port-based classification fallback."""
        ports = {dst_port, src_port}

        # Banking ports (P3)
        if any(p in [443, 5003, 8443] for p in ports):
            return "P3"

        # Voice ports (P2)
        if any(p in [5060, 5061, 5002, 3478, 3479] for p in ports):
            return "P2"

        # Bulk ports (P0)
        if any(p in [20, 21, 22, 5000, 8080] for p in ports):
            return "P0"

        # Default to web (P1)
        return "P1"

    def apply(
        self, flows: List[Dict], link_predictions: Dict[str, Dict]
    ) -> List[PolicyDecision]:
        """
        Apply policies to flows based on classification and predictions.

        Args:
            flows: List of flows (from PacketCapture or OVS)
            link_predictions: Dict of link predictions with congestion info

        Returns:
            List of policy decisions made
        """
        decisions = []

        # Analyze link states
        congestion_state = self._analyze_congestion(link_predictions)

        for flow in flows:
            # Skip flows without IP addresses
            if not flow.get("src_ip") or not flow.get("dst_ip"):
                continue

            # Classify if not already classified
            if "priority" not in flow:
                flow["priority"] = self.classify_flow(flow)

            # Make decision
            decision = self._decide_action(flow, congestion_state)
            if decision:
                decisions.append(decision)
                self.active_decisions[decision.flow_id] = decision
                self.stats["decisions_made"] += 1

                # Execute the decision
                self._execute_decision(decision, flow)

        return decisions

    def _analyze_congestion(self, link_predictions: Dict[str, Dict]) -> Dict[str, bool]:
        """
        Analyze link predictions to determine congestion state.

        Returns:
            Dict with congestion state flags
        """
        any_congested = False
        any_predicted = False
        congested_links = []

        for link_id, pred in link_predictions.items():
            utilization = pred.get("current_utilization", 0.0)
            is_congested = (
                pred.get("is_congested", False)
                or utilization >= self.CONGESTION_THRESHOLD
            )
            is_predicted = (
                pred.get("predicted_congestion", False)
                or utilization >= self.PREDICTION_THRESHOLD
            )

            if is_congested:
                any_congested = True
                congested_links.append(link_id)
            if is_predicted:
                any_predicted = True

        return {
            "any_congested": any_congested,
            "any_predicted": any_predicted,
            "congested_links": congested_links,
        }

    def _decide_action(
        self, flow: Dict, congestion_state: Dict[str, Any]
    ) -> Optional[PolicyDecision]:
        """
        Decide what action to take for a flow.

        Args:
            flow: Flow dictionary
            congestion_state: Congestion analysis results

        Returns:
            PolicyDecision or None if no action needed
        """
        flow_id = flow.get("flow_id", f"{flow.get('src_ip')}:{flow.get('src_port')}")
        priority = flow.get("priority", "P1")

        any_congested = congestion_state.get("any_congested", False)
        any_predicted = congestion_state.get("any_predicted", False)

        queue_id = self.PRIORITY_QUEUES.get(priority, 2)

        # Base parameters
        base_params = {
            "queue": queue_id,
            "src_ip": flow.get("src_ip", ""),
            "dst_ip": flow.get("dst_ip", ""),
            "src_port": flow.get("src_port", 0),
            "dst_port": flow.get("dst_port", 0),
            "protocol": flow.get("protocol", 6),
        }

        # P3 (Banking): Proactive reroute on predicted congestion
        if priority == "P3":
            if any_predicted:
                return PolicyDecision(
                    flow_id=str(flow_id),
                    action=Action.REROUTE,
                    priority=priority,
                    reason="Banking traffic proactively rerouted (predicted congestion)",
                    parameters={**base_params, "path": PathChoice.PATH_B},
                    src_ip=base_params["src_ip"],
                    dst_ip=base_params["dst_ip"],
                    src_port=base_params["src_port"],
                    dst_port=base_params["dst_port"],
                    protocol=base_params["protocol"],
                )
            return PolicyDecision(
                flow_id=str(flow_id),
                action=Action.PRIORITY_QUEUE,
                priority=priority,
                reason="Banking traffic gets highest priority queue",
                parameters=base_params,
                src_ip=base_params["src_ip"],
                dst_ip=base_params["dst_ip"],
                src_port=base_params["src_port"],
                dst_port=base_params["dst_port"],
                protocol=base_params["protocol"],
            )

        # P2 (Voice): Always priority queue, no rerouting (jitter sensitive)
        elif priority == "P2":
            return PolicyDecision(
                flow_id=str(flow_id),
                action=Action.PRIORITY_QUEUE,
                priority=priority,
                reason="Voice traffic gets low-jitter queue",
                parameters=base_params,
                src_ip=base_params["src_ip"],
                dst_ip=base_params["dst_ip"],
                src_port=base_params["src_port"],
                dst_port=base_params["dst_port"],
                protocol=base_params["protocol"],
            )

        # P0 (Bulk): Reactive reroute when actually congested
        elif priority == "P0":
            if any_congested:
                return PolicyDecision(
                    flow_id=str(flow_id),
                    action=Action.REROUTE,
                    priority=priority,
                    reason="Bulk traffic reactively rerouted (congestion detected)",
                    parameters={**base_params, "path": PathChoice.PATH_B},
                    src_ip=base_params["src_ip"],
                    dst_ip=base_params["dst_ip"],
                    src_port=base_params["src_port"],
                    dst_port=base_params["dst_port"],
                    protocol=base_params["protocol"],
                )
            return PolicyDecision(
                flow_id=str(flow_id),
                action=Action.PRIORITY_QUEUE,
                priority=priority,
                reason="Bulk traffic gets lowest priority queue",
                parameters=base_params,
                src_ip=base_params["src_ip"],
                dst_ip=base_params["dst_ip"],
                src_port=base_params["src_port"],
                dst_port=base_params["dst_port"],
                protocol=base_params["protocol"],
            )

        # P1 (Web): Best effort
        return PolicyDecision(
            flow_id=str(flow_id),
            action=Action.ALLOW,
            priority=priority,
            reason="Web traffic gets best-effort queue",
            parameters=base_params,
            src_ip=base_params["src_ip"],
            dst_ip=base_params["dst_ip"],
            src_port=base_params["src_port"],
            dst_port=base_params["dst_port"],
            protocol=base_params["protocol"],
        )

    def _execute_decision(self, decision: PolicyDecision, flow: Dict) -> None:
        """
        Execute a policy decision using the QoS Enforcer.

        Args:
            decision: The policy decision to execute
            flow: Original flow data
        """
        # Determine which switch to configure
        # In our topology, flows originate from s1 (hosts h1, h2)
        switch = self._determine_switch(decision.src_ip)

        # Create flow match
        match = FlowMatch(
            src_ip=decision.src_ip,
            dst_ip=decision.dst_ip,
            src_port=decision.src_port,
            dst_port=decision.dst_port,
            protocol=decision.protocol,
        )

        queue_id = decision.parameters.get("queue", 2)

        if decision.action == Action.REROUTE:
            path = decision.parameters.get("path", PathChoice.PATH_B)
            success = self.qos_enforcer.reroute_flow(switch, match, path, queue_id)
            if success:
                self.stats["reroutes_executed"] += 1
                logger.info(f"REROUTE: {decision}")

        elif decision.action == Action.PRIORITY_QUEUE:
            success = self.qos_enforcer.install_qos_rule(switch, match, queue_id)
            if success:
                self.stats["qos_rules_applied"] += 1
                logger.info(f"QOS: {decision}")

        elif decision.action == Action.ALLOW:
            # Best effort - still assign to appropriate queue
            success = self.qos_enforcer.install_qos_rule(switch, match, queue_id)
            if success:
                self.stats["qos_rules_applied"] += 1
                logger.debug(f"ALLOW: {decision}")

        elif decision.action == Action.THROTTLE:
            # Throttling is handled by queue rate limits (already in setup_qos.sh)
            success = self.qos_enforcer.install_qos_rule(switch, match, queue_id)
            if success:
                self.stats["qos_rules_applied"] += 1
                logger.info(f"THROTTLE: {decision}")

    def _determine_switch(self, src_ip: str) -> str:
        """
        Determine which edge switch handles traffic from this source IP.

        Args:
            src_ip: Source IP address

        Returns:
            Switch name (s1 or s4)
        """
        # In our topology:
        # s1 connects to h1 (10.0.0.1), h2 (10.0.0.2)
        # s4 connects to h3 (10.0.0.3), h4 (10.0.0.4)
        if src_ip in ("10.0.0.1", "10.0.0.2"):
            return "s1"
        elif src_ip in ("10.0.0.3", "10.0.0.4"):
            return "s4"
        else:
            # Default to s1 for unknown sources
            return "s1"

    def get_stats(self) -> Dict[str, Any]:
        """Get policy engine statistics."""
        enforcer_stats = self.qos_enforcer.get_stats()
        classifier_stats = self.sni_classifier.get_stats()

        # Include ML classifier stats if available
        ml_stats = {}
        if self.ml_classifier is not None:
            ml_stats = self.ml_classifier.get_stats()

        return {
            "policy_engine": self.stats,
            "qos_enforcer": enforcer_stats,
            "sni_classifier": classifier_stats,
            "ml_classifier": ml_stats,
            "ml_enabled": self.ml_classifier is not None,
        }

    def cleanup(self) -> None:
        """Cleanup all installed rules."""
        count = self.qos_enforcer.clear_all_rules()
        logger.info(f"Policy engine cleanup: removed {count} rules")


if __name__ == "__main__":
    # Test the policy engine
    import json

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Policy Engine Test (Dry Run)")
    print("=" * 70)

    engine = PolicyEngine(dry_run=True)

    # Test flows with ML features for proper classification
    test_flows = [
        {
            "flow_id": "banking-1",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.3",
            "src_port": 50000,
            "dst_port": 443,
            "protocol": 6,
            "sni": "netbanking.hdfcbank.com",
            # ML features (banking-like: small packets, bursty)
            "packet_count": 50,
            "byte_count": 15000,
            "duration_sec": 2.0,
            "bytes_per_packet": 300,
            "packets_per_sec": 25,
            "bytes_per_sec": 7500,
            "pkt_len_min": 64,
            "pkt_len_max": 500,
            "pkt_len_mean": 300,
            "pkt_len_std": 100,
            "iat_mean": 0.04,
            "iat_std": 0.02,
        },
        {
            "flow_id": "voice-1",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.4",
            "src_port": 50001,
            "dst_port": 5060,
            "protocol": 17,
            "sni": "meet.google.com",
            # ML features (voice-like: consistent small packets)
            "packet_count": 500,
            "byte_count": 100000,
            "duration_sec": 30.0,
            "bytes_per_packet": 200,
            "packets_per_sec": 16.7,
            "bytes_per_sec": 3333,
            "pkt_len_min": 160,
            "pkt_len_max": 240,
            "pkt_len_mean": 200,
            "pkt_len_std": 20,
            "iat_mean": 0.02,
            "iat_std": 0.005,
        },
        {
            "flow_id": "bulk-1",
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.3",
            "src_port": 50002,
            "dst_port": 5000,
            "protocol": 6,
            "sni": "files.dropbox.com",
            # ML features (bulk: large packets, sustained)
            "packet_count": 2000,
            "byte_count": 3000000,
            "duration_sec": 30.0,
            "bytes_per_packet": 1500,
            "packets_per_sec": 66.7,
            "bytes_per_sec": 100000,
            "pkt_len_min": 1200,
            "pkt_len_max": 1500,
            "pkt_len_mean": 1450,
            "pkt_len_std": 100,
            "iat_mean": 0.0001,
            "iat_std": 0.00005,
        },
        {
            "flow_id": "web-1",
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.4",
            "src_port": 50003,
            "dst_port": 80,
            "protocol": 6,
            "sni": "docs.google.com",
            # ML features (web: mixed packets)
            "packet_count": 100,
            "byte_count": 80000,
            "duration_sec": 5.0,
            "bytes_per_packet": 800,
            "packets_per_sec": 20,
            "bytes_per_sec": 16000,
            "pkt_len_min": 64,
            "pkt_len_max": 1500,
            "pkt_len_mean": 800,
            "pkt_len_std": 400,
            "iat_mean": 0.05,
            "iat_std": 0.1,
        },
    ]

    # Test predictions - simulate predicted congestion on Path A
    test_predictions = {
        "s1:3": {
            "current_utilization": 0.75,
            "is_congested": False,
            "predicted_congestion": True,
        },
        "s1:4": {
            "current_utilization": 0.30,
            "is_congested": False,
            "predicted_congestion": False,
        },
    }

    print("\n1. Classification Test:")
    print("-" * 50)
    for flow in test_flows:
        priority = engine.classify_flow(flow)
        print(f"  {flow['sni']}: {priority}")

    print("\n2. Policy Application (Predicted Congestion on Path A):")
    print("-" * 50)
    decisions = engine.apply(test_flows, test_predictions)
    for d in decisions:
        print(f"  {d}")

    print("\n3. Simulate Actual Congestion:")
    print("-" * 50)
    test_predictions["s1:3"]["is_congested"] = True
    test_predictions["s1:3"]["current_utilization"] = 0.85

    decisions = engine.apply(test_flows, test_predictions)
    for d in decisions:
        if d.action == Action.REROUTE:
            print(f"  {d}")

    print("\n4. Statistics:")
    print("-" * 50)
    print(json.dumps(engine.get_stats(), indent=2, default=str))

    print("\n5. Cleanup:")
    print("-" * 50)
    engine.cleanup()

    print("\nTest complete!")
