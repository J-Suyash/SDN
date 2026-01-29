import unittest
import time
from scapy.all import IP, TCP, UDP, Ether
from collector.packet_capture import PacketCapture, FlowStats


class TestPacketCaptureLogic(unittest.TestCase):
    def setUp(self):
        self.pc = PacketCapture(interface="test0")

    def test_tcp_flow_aggregation(self):
        """Test that TCP packets between same pair are aggregated into one flow."""
        # Packet 1: A -> B
        pkt1 = IP(src="192.168.1.10", dst="10.0.0.1") / TCP(sport=12345, dport=80)
        # Packet 2: B -> A
        pkt2 = IP(src="10.0.0.1", dst="192.168.1.10") / TCP(sport=80, dport=12345)

        # Inject packets
        self.pc._process_packet(pkt1)
        # Simulate small delay for IAT
        time.sleep(0.01)
        # Manually adjust timestamp for pkt2 to ensure positive IAT if we were using real timestamps
        # But _process_packet uses pkt.time if available. Scapy packets created this way might default to now.
        # Let's verify pkt.time behavior or mock it.
        pkt2.time = pkt1.time + 0.05
        self.pc._process_packet(pkt2)

        stats = self.pc.get_flow_stats()
        self.assertEqual(len(stats), 1, "Should be 1 aggregated flow")

        flow = stats[0]
        self.assertEqual(flow["packet_count"], 2)
        self.assertEqual(flow["protocol"], 6)  # TCP
        self.assertAlmostEqual(flow["duration"], 0.05, places=2)

        # Check canonical IPs (sorted)
        expected_src = "10.0.0.1"
        expected_dst = "192.168.1.10"
        self.assertEqual(flow["src_ip"], expected_src)
        self.assertEqual(flow["dst_ip"], expected_dst)

    def test_udp_flow(self):
        """Test UDP flow stats."""
        pkt = (
            IP(src="10.0.0.5", dst="10.0.0.6")
            / UDP(sport=5000, dport=5000)
            / ("X" * 100)
        )
        self.pc._process_packet(pkt)

        stats = self.pc.get_flow_stats()
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["protocol"], 17)  # UDP
        self.assertGreater(stats[0]["byte_count"], 100)  # Header + payload

    def test_pruning(self):
        """Test that old flows are pruned."""
        self.pc.idle_timeout = 0.1
        pkt = IP(src="1.1.1.1", dst="2.2.2.2") / TCP()
        self.pc._process_packet(pkt)

        self.assertEqual(len(self.pc.flows), 1)
        time.sleep(0.2)
        self.pc._prune_flows()
        self.assertEqual(len(self.pc.flows), 0, "Flow should be pruned after timeout")


if __name__ == "__main__":
    unittest.main()
