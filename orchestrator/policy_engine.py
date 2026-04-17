import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from orchestrator.types import PRIORITY_QUEUES
from orchestrator.sni_classifier import SNIClassifier
from orchestrator.qos_enforcer import QoSEnforcer, FlowMatch, PathChoice

try:
    from orchestrator.ml_classifier import MLTrafficClassifier, get_classifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger("policy_engine")


class Action(Enum):
    ALLOW = "allow"
    THROTTLE = "throttle"
    REROUTE = "reroute"
    PRIORITY_QUEUE = "priority_queue"


@dataclass
class PolicyDecision:
    flow_id: str
    action: Action
    priority: str
    reason: str
    parameters: Dict[str, Any]
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 6

    def __str__(self):
        return f"Flow {self.flow_id} ({self.priority}): {self.action.value} - {self.reason}"


class PolicyEngine:
    CONGESTION_THRESHOLD = 0.80
    PREDICTION_THRESHOLD = 0.70

    def __init__(self, dry_run: bool = False, use_ml: bool = True):
        self.sni_classifier = SNIClassifier()
        self.qos_enforcer = QoSEnforcer(dry_run=dry_run)
        self.dry_run = dry_run
        self.use_ml = use_ml

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

        self.active_decisions: Dict[str, PolicyDecision] = {}
        self.stats = {
            "decisions_made": 0,
            "qos_rules_applied": 0,
            "reroutes_executed": 0,
            "classification_by_ml": 0,
            "classification_by_sni": 0,
            "classification_by_port": 0,
        }

    def classify_flow(self, flow: Dict[str, Any]) -> str:
        # ML classifier handles the full cascade: ML -> SNI -> port
        if self.ml_classifier is not None:
            result = self.ml_classifier.classify(flow)
            if result.method == "ml":
                self.stats["classification_by_ml"] += 1
            elif result.method == "sni":
                self.stats["classification_by_sni"] += 1
            else:
                self.stats["classification_by_port"] += 1
            return result.priority

        # Without ML model: SNI -> port fallback
        sni = flow.get("sni", "")
        if sni and sni.lower() not in ("unknown", "none", ""):
            priority = self.sni_classifier.classify(sni)
            if priority != "P1":
                self.stats["classification_by_sni"] += 1
                return priority

        dst_port = flow.get("dst_port", 0)
        src_port = flow.get("src_port", 0)
        priority = self._classify_by_port(dst_port, src_port)
        self.stats["classification_by_port"] += 1
        return priority

    def _classify_by_port(self, dst_port: int, src_port: int) -> str:
        ports = {dst_port, src_port}
        if any(p in [443, 5003, 8443] for p in ports):
            return "P3"
        if any(p in [5060, 5061, 5002, 3478, 3479] for p in ports):
            return "P2"
        if any(p in [20, 21, 22, 5000, 8080] for p in ports):
            return "P0"
        return "P1"

    def apply(
        self, flows: List[Dict], link_predictions: Dict[str, Dict],
    ) -> List[PolicyDecision]:
        decisions = []
        congestion_state = self._analyze_congestion(link_predictions)

        for flow in flows:
            if not flow.get("src_ip") or not flow.get("dst_ip"):
                continue
            if "priority" not in flow:
                flow["priority"] = self.classify_flow(flow)

            decision = self._decide_action(flow, congestion_state)
            if decision:
                decisions.append(decision)
                self.active_decisions[decision.flow_id] = decision
                self.stats["decisions_made"] += 1
                self._execute_decision(decision)

        return decisions

    def _analyze_congestion(self, link_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        any_congested = False
        any_predicted = False
        congested_links = []

        for link_id, pred in link_predictions.items():
            utilization = pred.get("current_utilization", 0.0)
            is_congested = pred.get("is_congested", False) or utilization >= self.CONGESTION_THRESHOLD
            is_predicted = pred.get("predicted_congestion", False) or utilization >= self.PREDICTION_THRESHOLD

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
        self, flow: Dict, congestion_state: Dict[str, Any],
    ) -> Optional[PolicyDecision]:
        flow_id = flow.get("flow_id", f"{flow.get('src_ip')}:{flow.get('src_port')}")
        priority = flow.get("priority", "P1")
        any_congested = congestion_state.get("any_congested", False)
        any_predicted = congestion_state.get("any_predicted", False)
        queue_id = PRIORITY_QUEUES.get(priority, 2)

        base = {
            "queue": queue_id,
            "src_ip": flow.get("src_ip", ""),
            "dst_ip": flow.get("dst_ip", ""),
            "src_port": flow.get("src_port", 0),
            "dst_port": flow.get("dst_port", 0),
            "protocol": flow.get("protocol", 6),
        }

        def _make(action, reason, extra_params=None):
            params = {**base, **(extra_params or {})}
            return PolicyDecision(
                flow_id=str(flow_id), action=action, priority=priority,
                reason=reason, parameters=params,
                src_ip=base["src_ip"], dst_ip=base["dst_ip"],
                src_port=base["src_port"], dst_port=base["dst_port"],
                protocol=base["protocol"],
            )

        # P3 (Banking): Proactive reroute on predicted congestion
        if priority == "P3":
            if any_predicted:
                return _make(Action.REROUTE, "Banking traffic proactively rerouted (predicted congestion)", {"path": PathChoice.PATH_B})
            return _make(Action.PRIORITY_QUEUE, "Banking traffic gets highest priority queue")

        if priority == "P2":
            return _make(Action.PRIORITY_QUEUE, "Voice traffic gets low-jitter queue")

        # P0 (Bulk): Reactive reroute when actually congested
        if priority == "P0":
            if any_congested:
                return _make(Action.REROUTE, "Bulk traffic reactively rerouted (congestion detected)", {"path": PathChoice.PATH_B})
            return _make(Action.PRIORITY_QUEUE, "Bulk traffic gets lowest priority queue")

        return _make(Action.ALLOW, "Web traffic gets best-effort queue")

    def _execute_decision(self, decision: PolicyDecision) -> None:
        switch = self._determine_switch(decision.src_ip)
        match = FlowMatch(
            src_ip=decision.src_ip, dst_ip=decision.dst_ip,
            src_port=decision.src_port, dst_port=decision.dst_port,
            protocol=decision.protocol,
        )
        queue_id = decision.parameters.get("queue", 2)

        if decision.action == Action.REROUTE:
            path = decision.parameters.get("path", PathChoice.PATH_B)
            if self.qos_enforcer.reroute_flow(switch, match, path, queue_id):
                self.stats["reroutes_executed"] += 1
        else:
            if self.qos_enforcer.install_qos_rule(switch, match, queue_id):
                self.stats["qos_rules_applied"] += 1

    def _determine_switch(self, src_ip: str) -> str:
        if src_ip in ("10.0.0.1", "10.0.0.2"):
            return "s1"
        elif src_ip in ("10.0.0.3", "10.0.0.4"):
            return "s4"
        return "s1"

    def get_stats(self) -> Dict[str, Any]:
        ml_stats = {}
        if self.ml_classifier is not None:
            ml_stats = self.ml_classifier.get_stats()

        return {
            "policy_engine": self.stats,
            "qos_enforcer": self.qos_enforcer.get_stats(),
            "sni_classifier": self.sni_classifier.get_stats(),
            "ml_classifier": ml_stats,
            "ml_enabled": self.ml_classifier is not None,
        }

    def cleanup(self) -> None:
        count = self.qos_enforcer.clear_all_rules()
        logger.info(f"Policy engine cleanup: removed {count} rules")
