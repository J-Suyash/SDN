"""Tests for the QoS enforcer (OVS command generation)."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.qos_enforcer import (
    QoSEnforcer, FlowMatch, PathChoice, InstalledRule,
)


class TestFlowMatch(unittest.TestCase):
    """Test the FlowMatch dataclass."""

    def test_to_ovs_match_tcp(self):
        match = FlowMatch(
            src_ip="10.0.0.1", dst_ip="10.0.0.4",
            src_port=12345, dst_port=443, protocol=6,
        )
        result = match.to_ovs_match()
        self.assertIn("tcp", result)
        self.assertIn("nw_src=10.0.0.1", result)
        self.assertIn("nw_dst=10.0.0.4", result)
        self.assertIn("tp_src=12345", result)
        self.assertIn("tp_dst=443", result)

    def test_to_ovs_match_udp(self):
        match = FlowMatch(
            src_ip="10.0.0.1", dst_ip="10.0.0.4",
            src_port=5002, dst_port=5002, protocol=17,
        )
        result = match.to_ovs_match()
        self.assertIn("udp", result)

    def test_to_ovs_match_no_ports(self):
        match = FlowMatch(
            src_ip="10.0.0.1", dst_ip="10.0.0.4",
            src_port=0, dst_port=0, protocol=6,
        )
        result = match.to_ovs_match()
        self.assertNotIn("tp_src=0", result)
        self.assertNotIn("tp_dst=0", result)

    def test_to_reverse_match(self):
        match = FlowMatch(
            src_ip="10.0.0.1", dst_ip="10.0.0.4",
            src_port=12345, dst_port=443, protocol=6,
        )
        reverse = match.to_reverse_match()
        self.assertEqual(reverse.src_ip, "10.0.0.4")
        self.assertEqual(reverse.dst_ip, "10.0.0.1")
        self.assertEqual(reverse.src_port, 443)
        self.assertEqual(reverse.dst_port, 12345)
        self.assertEqual(reverse.protocol, 6)

    def test_flow_id_consistent(self):
        match1 = FlowMatch("10.0.0.1", "10.0.0.4", 12345, 443, 6)
        match2 = FlowMatch("10.0.0.1", "10.0.0.4", 12345, 443, 6)
        self.assertEqual(match1.flow_id, match2.flow_id)

    def test_flow_id_changes_with_ports(self):
        match1 = FlowMatch("10.0.0.1", "10.0.0.4", 12345, 443, 6)
        match2 = FlowMatch("10.0.0.1", "10.0.0.4", 99999, 443, 6)
        self.assertNotEqual(match1.flow_id, match2.flow_id)

    def test_protocol_name(self):
        self.assertEqual(FlowMatch("a", "b", 0, 0, 6).protocol_name, "tcp")
        self.assertEqual(FlowMatch("a", "b", 0, 0, 17).protocol_name, "udp")


class TestQoSEnforcer(unittest.TestCase):
    """Test QoS enforcer logic (dry-run mode, no actual OVS calls)."""

    def setUp(self):
        self.enforcer = QoSEnforcer(dry_run=True)
        self.match = FlowMatch(
            src_ip="10.0.0.1", dst_ip="10.0.0.4",
            src_port=12345, dst_port=443, protocol=6,
        )

    def test_install_qos_rule_dry_run(self):
        result = self.enforcer.install_qos_rule("s1", self.match, queue_id=0)
        self.assertTrue(result)
        stats = self.enforcer.get_stats()
        self.assertGreaterEqual(stats["rules_installed"], 1)

    def test_double_install_same_rule_is_noop(self):
        self.enforcer.install_qos_rule("s1", self.match, queue_id=0)
        stats_before = self.enforcer.get_stats()["rules_installed"]
        self.enforcer.install_qos_rule("s1", self.match, queue_id=0)
        stats_after = self.enforcer.get_stats()["rules_installed"]
        # Should not re-install identical rule
        self.assertEqual(stats_before, stats_after)

    def test_reroute_flow(self):
        result = self.enforcer.reroute_flow(
            "s1", self.match, PathChoice.PATH_B, queue_id=0,
        )
        self.assertTrue(result)
        stats = self.enforcer.get_stats()
        self.assertGreaterEqual(stats["reroutes"], 1)

    def test_delete_rule(self):
        self.enforcer.install_qos_rule("s1", self.match, queue_id=0)
        result = self.enforcer.delete_rule("s1", self.match)
        self.assertTrue(result)
        stats = self.enforcer.get_stats()
        self.assertGreaterEqual(stats["rules_deleted"], 1)

    def test_clear_reroute(self):
        self.enforcer.reroute_flow("s1", self.match, PathChoice.PATH_B, queue_id=0)
        result = self.enforcer.clear_reroute("s1", self.match)
        self.assertTrue(result)

    def test_cleanup_stale_rules(self):
        self.enforcer.install_qos_rule("s1", self.match, queue_id=0)
        import time
        # Rules installed just now should not be stale with max_age=300
        count = self.enforcer.cleanup_stale_rules(max_age_seconds=0)
        self.assertGreaterEqual(count, 1)

    def test_clear_all_rules(self):
        match2 = FlowMatch("10.0.0.2", "10.0.0.3", 80, 80, 6)
        self.enforcer.install_qos_rule("s1", self.match, queue_id=0)
        self.enforcer.install_qos_rule("s1", match2, queue_id=2)
        count = self.enforcer.clear_all_rules()
        self.assertGreaterEqual(count, 2)
        self.assertEqual(self.enforcer.get_stats()["active_rules"], 0)

    def test_path_choice_attributes(self):
        self.assertEqual(PathChoice.PATH_A.egress_port, 3)
        self.assertEqual(PathChoice.PATH_B.egress_port, 4)
        self.assertIn("s2", PathChoice.PATH_A.description)
        self.assertIn("s5", PathChoice.PATH_B.description)

    def test_stats_tracking(self):
        self.enforcer.install_qos_rule("s1", self.match, queue_id=0)
        self.enforcer.reroute_flow("s1", self.match, PathChoice.PATH_B, queue_id=0)
        self.enforcer.delete_rule("s1", self.match)
        stats = self.enforcer.get_stats()
        self.assertGreater(stats["rules_installed"], 0)
        self.assertGreater(stats["rules_deleted"], 0)
        self.assertGreater(stats["reroutes"], 0)


class TestFlowMatchFromDict(unittest.TestCase):
    """Test the flow_match_from_dict helper."""

    def test_from_dict_tcp(self):
        from orchestrator.qos_enforcer import flow_match_from_dict
        flow = {
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.4",
            "src_port": "12345",
            "dst_port": "443",
            "protocol": "tcp",
        }
        fm = flow_match_from_dict(flow)
        self.assertEqual(fm.src_ip, "10.0.0.1")
        self.assertEqual(fm.src_port, 12345)
        self.assertEqual(fm.protocol, 6)

    def test_from_dict_udp(self):
        from orchestrator.qos_enforcer import flow_match_from_dict
        flow = {
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.4",
            "src_port": 5002, "dst_port": 5002,
            "protocol": "udp",
        }
        fm = flow_match_from_dict(flow)
        self.assertEqual(fm.protocol, 17)


if __name__ == "__main__":
    unittest.main()
