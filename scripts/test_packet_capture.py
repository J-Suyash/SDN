"""
Test script for PacketCapture module.
"""

import time
import requests
import threading
import logging
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collector.packet_capture import PacketCapture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPacketCapture")


def generate_traffic():
    """Generate some local traffic."""
    time.sleep(2)
    logger.info("Generating traffic...")

    # 1. Ping
    os.system("ping -c 3 127.0.0.1 > /dev/null")

    # 2. HTTP request to self (might fail connection but generates packets)
    try:
        requests.get("http://127.0.0.1:8000", timeout=1)
    except:
        pass


def test_capture():
    # Use lo for safety
    interface = "lo"

    logger.info(f"Initializing capture on {interface}...")
    pc = PacketCapture(interface=interface)

    pc.start()

    # Run traffic generator in background
    t = threading.Thread(target=generate_traffic)
    t.start()
    t.join()

    time.sleep(2)  # Wait for processing

    stats = pc.get_flow_stats()
    logger.info(f"Captured {len(stats)} flows.")

    for flow in stats:
        print("--- Flow ---")
        print(f"Src: {flow['src_ip']}:{flow['src_port']}")
        print(f"Dst: {flow['dst_ip']}:{flow['dst_port']}")
        print(f"Proto: {flow['protocol']}")
        print(f"Pkts: {flow['total_packets']}, Bytes: {flow['total_bytes']}")
        print(f"Duration: {flow['duration']:.4f}s")
        print(f"IAT Mean: {flow['iat_mean']:.6f}")
        print(f"SNI: {flow['sni']}")

    pc.stop()


if __name__ == "__main__":
    test_capture()
