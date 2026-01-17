"""
Policy Engine for SDN Traffic Management
Maps classification and prediction outputs to network actions.
"""

import logging
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("policy_engine")


class Action(Enum):
    """Network policy actions"""

    ALLOW = "allow"
    THROTTLE = "throttle"
    REROUTE = "reroute"
    PRIORITY_QUEUE = "priority_queue"


@dataclass
class PolicyDecision:
    """A policy decision for a flow"""

    flow_id: int
    action: Action
    priority: str
    reason: str
    parameters: Dict[str, Any]

    def __str__(self):
        return f"Flow {self.flow_id} ({self.priority}): {self.action.value} - {self.reason}"


class PolicyEngine:
    """
    Applies network policies based on flow classification and congestion predictions.

    Priority Rules:
    - P3 (Banking): Always priority queue, reroute if congestion predicted
    - P2 (Voice): Priority queue, maintain low jitter
    - P1 (Web): Best effort
    - P0 (Bulk): Throttle under congestion, reroute first
    """

    # QoS queue mapping
    PRIORITY_QUEUES = {
        "P3": 0,  # Highest priority queue
        "P2": 1,
        "P1": 2,
        "P0": 3,  # Lowest priority queue
    }

    # Throttle rates under congestion (percentage of link capacity)
    THROTTLE_RATES = {
        "P3": 100,  # No throttling
        "P2": 90,
        "P1": 70,
        "P0": 30,  # Heavily throttled
    }

    def __init__(self):
        self.active_decisions: Dict[int, PolicyDecision] = {}

    def apply(
        self, flows: List[Dict], link_predictions: Dict[str, Dict]
    ) -> List[PolicyDecision]:
        """
        Apply policies to flows based on classification and predictions.

        Args:
            flows: List of classified flows with 'priority' field
            link_predictions: Dict of link predictions with congestion info

        Returns:
            List of policy decisions made
        """
        decisions = []

        # Check if any link is congested or predicted congested
        any_congested = any(
            p.get("is_congested", False) for p in link_predictions.values()
        )
        any_predicted_congestion = any(
            p.get("predicted_congestion", False) for p in link_predictions.values()
        )

        for flow in flows:
            flow_id = flow.get("flow_id", 0)
            priority = flow.get("priority", "P1")

            decision = self._decide_action(
                flow_id=flow_id,
                priority=priority,
                flow=flow,
                any_congested=any_congested,
                any_predicted_congestion=any_predicted_congestion,
            )

            decisions.append(decision)
            self.active_decisions[flow_id] = decision

            # Apply the decision
            self._execute_decision(decision)

        return decisions

    def _decide_action(
        self,
        flow_id: int,
        priority: str,
        flow: Dict,
        any_congested: bool,
        any_predicted_congestion: bool,
    ) -> PolicyDecision:
        """Decide what action to take for a flow"""

        if priority == "P3":
            # Banking: Always high priority
            if any_predicted_congestion:
                return PolicyDecision(
                    flow_id=flow_id,
                    action=Action.REROUTE,
                    priority=priority,
                    reason="Banking traffic avoiding predicted congestion",
                    parameters={
                        "queue": self.PRIORITY_QUEUES[priority],
                        "reroute": True,
                    },
                )
            return PolicyDecision(
                flow_id=flow_id,
                action=Action.PRIORITY_QUEUE,
                priority=priority,
                reason="Banking traffic gets highest priority",
                parameters={"queue": self.PRIORITY_QUEUES[priority]},
            )

        elif priority == "P2":
            # Voice: Priority queue, sensitive to jitter
            return PolicyDecision(
                flow_id=flow_id,
                action=Action.PRIORITY_QUEUE,
                priority=priority,
                reason="Voice traffic gets low-jitter queue",
                parameters={"queue": self.PRIORITY_QUEUES[priority]},
            )

        elif priority == "P0":
            # Bulk: Throttle first under congestion
            if any_congested:
                return PolicyDecision(
                    flow_id=flow_id,
                    action=Action.THROTTLE,
                    priority=priority,
                    reason="Bulk traffic throttled due to congestion",
                    parameters={
                        "queue": self.PRIORITY_QUEUES[priority],
                        "rate_limit": self.THROTTLE_RATES[priority],
                    },
                )
            elif any_predicted_congestion:
                return PolicyDecision(
                    flow_id=flow_id,
                    action=Action.REROUTE,
                    priority=priority,
                    reason="Bulk traffic rerouted to avoid predicted congestion",
                    parameters={
                        "queue": self.PRIORITY_QUEUES[priority],
                        "reroute": True,
                    },
                )

        # Default: Best effort (P1)
        return PolicyDecision(
            flow_id=flow_id,
            action=Action.ALLOW,
            priority=priority,
            reason="Best effort traffic",
            parameters={"queue": self.PRIORITY_QUEUES.get(priority, 2)},
        )

    def _execute_decision(self, decision: PolicyDecision) -> None:
        """
        Execute a policy decision on the network.

        MVP: Just logs the decision
        Phase 5+: Will execute OVS/Faucet commands
        """
        logger.debug(f"Executing: {decision}")

        # TODO: Implement actual enforcement in Phase 5
        # Examples:
        # - ovs-vsctl set port <port> qos=@qos
        # - Update Faucet ACLs
        # - Modify flow rules for rerouting

        if decision.action == Action.PRIORITY_QUEUE:
            queue = decision.parameters.get("queue", 2)
            logger.debug(f"Would assign flow {decision.flow_id} to queue {queue}")

        elif decision.action == Action.THROTTLE:
            rate = decision.parameters.get("rate_limit", 100)
            logger.debug(f"Would throttle flow {decision.flow_id} to {rate}%")

        elif decision.action == Action.REROUTE:
            logger.debug(f"Would reroute flow {decision.flow_id} to alternate path")

    def get_active_decisions(self) -> Dict[int, PolicyDecision]:
        """Get all active policy decisions"""
        return self.active_decisions.copy()


if __name__ == "__main__":
    # Test the policy engine
    logging.basicConfig(level=logging.DEBUG)

    engine = PolicyEngine()

    # Test flows
    flows = [
        {"flow_id": 1, "priority": "P3"},  # Banking
        {"flow_id": 2, "priority": "P2"},  # Voice
        {"flow_id": 3, "priority": "P1"},  # Web
        {"flow_id": 4, "priority": "P0"},  # Bulk
    ]

    # Test predictions
    predictions = {
        "s1:1": {"is_congested": False, "predicted_congestion": True},
    }

    print("Policy Engine Test:")
    print("-" * 60)

    decisions = engine.apply(flows, predictions)
    for d in decisions:
        print(d)
