"""Tests for the policy engine decision logic."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.policy_engine import PolicyEngine, Action, PolicyDecision
from orchestrator.qos_enforcer import PathChoice
from orchestrator.types import PRIORITY_QUEUES


class TestPolicyEngine(unittest.TestCase):
    """Test the policy engine's classification and decision logic."""

    def setUp(self):
        # Use dry_run=True to avoid actual OVS calls
        self.engine = PolicyEngine(dry_run=True, use_ml=False)

    def _make_flow(self, **kwargs):
        """Helper to create a flow dict with defaults."""
        flow = {
            "flow_id": "test_flow_1",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.4",
            "src_port": 12345,
            "dst_port": 443,
            "protocol": 6,
            "bytes": 50000,
            "packets": 100,
            "duration": 10.0,
        }
        flow.update(kwargs)
        return flow

    def _make_prediction(self, utilization=0.5, congested=False, predicted=False):
        """Helper to create a link prediction dict."""
        return {
            "s1:3": {
                "is_congested": congested,
                "predicted_congestion": predicted,
                "current_utilization": utilization,
                "trend": "up" if predicted else "stable",
            }
        }

    # --- Port-based classification tests ---

    def test_classify_port_443_is_p3(self):
        flow = self._make_flow(dst_port=443)
        self.assertEqual(self.engine.classify_flow(flow), "P3")

    def test_classify_port_5003_is_p3(self):
        flow = self._make_flow(dst_port=5003)
        self.assertEqual(self.engine.classify_flow(flow), "P3")

    def test_classify_port_5060_is_p2(self):
        flow = self._make_flow(dst_port=5060)
        self.assertEqual(self.engine.classify_flow(flow), "P2")

    def test_classify_port_5002_is_p2(self):
        flow = self._make_flow(dst_port=5002)
        self.assertEqual(self.engine.classify_flow(flow), "P2")

    def test_classify_port_5000_is_p0(self):
        flow = self._make_flow(dst_port=5000)
        self.assertEqual(self.engine.classify_flow(flow), "P0")

    def test_classify_port_80_is_p1(self):
        flow = self._make_flow(dst_port=80)
        self.assertEqual(self.engine.classify_flow(flow), "P1")

    def test_classify_unknown_port_defaults_to_p1(self):
        flow = self._make_flow(dst_port=9999)
        self.assertEqual(self.engine.classify_flow(flow), "P1")

    def test_classify_checks_both_ports(self):
        flow = self._make_flow(dst_port=9999, src_port=443)
        self.assertEqual(self.engine.classify_flow(flow), "P3")

    # --- Policy decision tests: P3 (Banking) ---

    def test_p3_normal_gets_priority_queue(self):
        flow = self._make_flow(dst_port=443)
        flow["priority"] = "P3"
        predictions = self._make_prediction(utilization=0.3)
        decisions = self.engine.apply([flow], predictions)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, Action.PRIORITY_QUEUE)
        self.assertIn("highest priority", decisions[0].reason)

    def test_p3_predicted_congestion_gets_proactive_reroute(self):
        flow = self._make_flow(dst_port=443)
        flow["priority"] = "P3"
        predictions = self._make_prediction(utilization=0.75, predicted=True)
        decisions = self.engine.apply([flow], predictions)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, Action.REROUTE)
        self.assertIn("proactively rerouted", decisions[0].reason)
        self.assertEqual(
            decisions[0].parameters.get("path"), PathChoice.PATH_B
        )

    def test_p3_queue_param_correct(self):
        flow = self._make_flow(dst_port=443)
        flow["priority"] = "P3"
        predictions = self._make_prediction(utilization=0.3)
        decisions = self.engine.apply([flow], predictions)
        self.assertEqual(decisions[0].parameters.get("queue"), 0)  # P3 queue

    # --- Policy decision tests: P2 (Voice) ---

    def test_p2_always_gets_priority_queue(self):
        flow = self._make_flow(dst_port=5060)
        flow["priority"] = "P2"
        predictions = self._make_prediction(utilization=0.9, congested=True)
        decisions = self.engine.apply([flow], predictions)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, Action.PRIORITY_QUEUE)
        self.assertIn("low-jitter", decisions[0].reason)
        self.assertEqual(decisions[0].parameters.get("queue"), 1)

    # --- Policy decision tests: P0 (Bulk) ---

    def test_p0_normal_gets_low_priority_queue(self):
        flow = self._make_flow(dst_port=5000)
        flow["priority"] = "P0"
        predictions = self._make_prediction(utilization=0.3)
        decisions = self.engine.apply([flow], predictions)
        self.assertEqual(decisions[0].action, Action.PRIORITY_QUEUE)
        self.assertIn("lowest priority", decisions[0].reason)

    def test_p0_congested_gets_reactive_reroute(self):
        flow = self._make_flow(dst_port=5000)
        flow["priority"] = "P0"
        predictions = self._make_prediction(utilization=0.85, congested=True)
        decisions = self.engine.apply([flow], predictions)
        self.assertEqual(decisions[0].action, Action.REROUTE)
        self.assertIn("reactively rerouted", decisions[0].reason)

    # --- Policy decision tests: P1 (Web) ---

    def test_p1_gets_best_effort(self):
        flow = self._make_flow(dst_port=80)
        flow["priority"] = "P1"
        predictions = self._make_prediction(utilization=0.5)
        decisions = self.engine.apply([flow], predictions)
        self.assertEqual(decisions[0].action, Action.ALLOW)
        self.assertIn("best-effort", decisions[0].reason)

    # --- Congestion analysis tests ---

    def test_analyze_congestion_detects_high_utilization(self):
        predictions = self._make_prediction(utilization=0.85, congested=True)
        state = self.engine._analyze_congestion(predictions)
        self.assertTrue(state["any_congested"])
        self.assertIn("s1:3", state["congested_links"])

    def test_analyze_congestion_no_congestion(self):
        predictions = self._make_prediction(utilization=0.3)
        state = self.engine._analyze_congestion(predictions)
        self.assertFalse(state["any_congested"])
        self.assertFalse(state["any_predicted"])

    # --- Edge cases ---

    def test_empty_flows_returns_no_decisions(self):
        decisions = self.engine.apply([], {})
        self.assertEqual(len(decisions), 0)

    def test_flow_missing_ips_skipped(self):
        flow = self._make_flow(src_ip="", dst_ip="")
        decisions = self.engine.apply([flow], {})
        self.assertEqual(len(decisions), 0)

    def test_stats_tracking(self):
        flow1 = self._make_flow(dst_port=443)
        flow2 = self._make_flow(dst_port=80, flow_id="flow_2")
        predictions = self._make_prediction(utilization=0.3)
        self.engine.apply([flow1, flow2], predictions)
        stats = self.engine.get_stats()
        self.assertGreaterEqual(stats["policy_engine"]["decisions_made"], 2)

    def test_qos_queue_mapping(self):
        for priority, expected_queue in [("P3", 0), ("P2", 1), ("P1", 2), ("P0", 3)]:
            with self.subTest(priority=priority):
                self.assertEqual(PRIORITY_QUEUES[priority], expected_queue)

    def test_multiple_flows_different_classes(self):
        flows = [
            self._make_flow(dst_port=443, flow_id="banking"),
            self._make_flow(dst_port=5060, flow_id="voice"),
            self._make_flow(dst_port=80, flow_id="web"),
            self._make_flow(dst_port=5000, flow_id="bulk"),
        ]
        predictions = self._make_prediction(utilization=0.85, congested=True)
        decisions = self.engine.apply(flows, predictions)

        # P3 gets proactive reroute on predicted OR priority queue
        # P2 gets priority queue
        # P1 gets allow
        # P0 gets reactive reroute
        actions = {d.flow_id: d.action for d in decisions}
        self.assertIn("banking", actions)
        self.assertIn("voice", actions)
        self.assertIn("web", actions)
        self.assertIn("bulk", actions)


class TestPolicyEngineWithML(unittest.TestCase):
    """Test the policy engine when ML classifier is available."""

    def setUp(self):
        self.engine = PolicyEngine(dry_run=True, use_ml=True)

    def test_engine_initializes_with_ml(self):
        """Should not crash when initializing with ML enabled."""
        self.assertIsNotNone(self.engine)


class TestPolicyDecision(unittest.TestCase):
    """Test the PolicyDecision dataclass."""

    def test_decision_str(self):
        d = PolicyDecision(
            flow_id="test1",
            action=Action.REROUTE,
            priority="P3",
            reason="proactive reroute",
            parameters={"path": PathChoice.PATH_B, "queue": 0},
            src_ip="10.0.0.1",
            dst_ip="10.0.0.4",
            src_port=443,
            dst_port=54321,
            protocol=6,
        )
        s = str(d)
        self.assertIn("test1", s)
        self.assertIn("P3", s)
        self.assertIn("reroute", s)


if __name__ == "__main__":
    unittest.main()
