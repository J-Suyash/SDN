import unittest
import os
import time
import shutil
import tempfile
from scapy.all import IP, TCP, Ether
from collector.packet_capture import PacketCapture, FlowStats
from collector.build_datasets import DatasetBuilder, label_flow_by_port


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.builder = DatasetBuilder(self.tmpdir)
        self.pc = PacketCapture(interface="test0")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        self.pc.stop()

    def test_end_to_end_flow(self):
        # 1. Simulate Packet Capture
        # Pkt1: HTTPS Client Hello (Simulated)
        # We won't build a full TLS packet here as it's complex,
        # but we'll verify basic TCP stats flow through.
        pkt1 = IP(src="192.168.1.100", dst="1.1.1.1") / TCP(sport=50000, dport=443)
        pkt1.time = time.time()
        self.pc._process_packet(pkt1)

        pkt2 = IP(src="1.1.1.1", dst="192.168.1.100") / TCP(sport=443, dport=50000)
        pkt2.time = pkt1.time + 0.1
        self.pc._process_packet(pkt2)

        # 2. Get Stats
        stats = self.pc.get_flow_stats()
        self.assertEqual(len(stats), 1)
        flow_stat = stats[0]

        # 3. Simulate Orchestrator Logic
        # Labeling
        # Note: flow_stat ports are canonical, so we must check both
        label = label_flow_by_port(flow_stat["dst_port"], flow_stat["src_port"])
        self.assertEqual(label, "P3")

        # Build Record
        record = self.builder.build_flow_record(flow_stat, label=label)

        # Verify Record Fields
        self.assertEqual(record.packet_count, 2)
        self.assertEqual(record.protocol, 6)
        self.assertEqual(
            record.pkt_len_min, 40
        )  # TCP header + IP header = 20+20=40 (approx)

        # 4. Add to Dataset
        self.builder.add_flow(record)

        # 5. Verify CSV
        with open(self.builder.flows_file, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Header + 1 row
            self.assertIn("P3", lines[1])
            self.assertIn("1.1.1.1", lines[1])


if __name__ == "__main__":
    unittest.main()
