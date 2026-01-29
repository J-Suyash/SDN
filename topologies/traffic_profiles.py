"""
Traffic Generation Profiles for SDN ML Project
Generates various traffic patterns for training and testing.
"""

import subprocess
import time
import threading
import random
import logging
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriorityClass(Enum):
    """Traffic priority classes"""

    P0_BULK = 0  # Bulk/Background - lowest priority
    P1_WEB = 1  # Office/Web - best effort
    P2_VOICE = 2  # Voice/Video - low jitter
    P3_BANKING = 3  # Banking/Payment - highest priority


@dataclass
class TrafficProfile:
    """Configuration for a traffic flow"""

    name: str
    priority: PriorityClass
    protocol: str  # 'tcp' or 'udp'
    bandwidth: Optional[str] = None  # e.g., '1M', '10M'
    duration: int = 30  # seconds
    port: int = 5001
    description: str = ""


# Predefined traffic profiles
TRAFFIC_PROFILES = {
    # P3: Banking - simulated with specific port
    "banking": TrafficProfile(
        name="banking",
        priority=PriorityClass.P3_BANKING,
        protocol="tcp",
        bandwidth="1M",
        duration=10,
        port=5003,
        description="Banking/Payment traffic (P3)",
    ),
    # P2: Voice - low bandwidth UDP
    "voice": TrafficProfile(
        name="voice",
        priority=PriorityClass.P2_VOICE,
        protocol="udp",
        bandwidth="64k",
        duration=60,
        port=5002,
        description="Voice/VoIP traffic (P2)",
    ),
    # P1: Web - bursty TCP
    "web": TrafficProfile(
        name="web",
        priority=PriorityClass.P1_WEB,
        protocol="tcp",
        bandwidth="2M",
        duration=30,
        port=5001,
        description="Web/Office traffic (P1)",
    ),
    # P0: Bulk - high bandwidth TCP
    "bulk": TrafficProfile(
        name="bulk",
        priority=PriorityClass.P0_BULK,
        protocol="tcp",
        bandwidth="10M",
        duration=120,
        port=5000,
        description="Bulk download traffic (P0)",
    ),
}


def run_iperf_server(host_name: str, port: int, udp: bool = False) -> subprocess.Popen:
    """
    Start iperf server on a host via Mininet CLI wrapper 'mn'.
    Requires Mininet topology to be running.
    """
    # Using 'm' utility if available, or assume 'mn exec' or similar
    # For host-based Mininet, we can usually use `mx` or just execute in the namespace if we knew the pid.
    # But since we are running outside, the best way is usually via the `mininet.util.custom` or just assume
    # we can run `sudo mn -c` is destructive.

    # NOTE: The reliable way to control a running Mininet from outside is using the
    # Mininet API or sending commands to the CLI stdin if it was started as a subprocess.

    # Since we can't easily attach to a running `mn` CLI from another process without
    # setup, we will change strategy: The *topology script* should import and run this.
    pass


def generate_traffic_internal(net, profile: TrafficProfile, src_name: str, dst_ip: str):
    """
    Generate traffic using the Mininet object directly.
    """
    src_host = net.get(src_name)

    logger.info(f"Generating {profile.name} traffic: {profile.description}")

    # Build iperf command
    cmd = f"iperf -c {dst_ip} -p {profile.port} -t {profile.duration} -i 1"
    if profile.protocol == "udp":
        cmd += " -u"
    if profile.bandwidth:
        cmd += f" -b {profile.bandwidth}"

    # Run in background
    src_host.cmd(cmd + " &")


def run_9am_burst_scenario(net, hosts: List[str], base_ip: str = "10.0.0") -> None:
    """Simulate 9AM office login burst using Mininet object."""
    logger.info("Starting 9AM burst scenario")

    # Phase 1: Login burst (0-60s)
    logger.info("Phase 1: Login burst")
    for i in range(5):  # Reduced for demo speed
        host_name = random.choice(hosts)
        generate_traffic_internal(
            net, TRAFFIC_PROFILES["web"], host_name, f"{base_ip}.3"
        )  # Target h3
        time.sleep(random.uniform(1, 3))

    logger.info("9AM burst scenario complete")


def run_banking_priority_demo(
    net, banking_host: str, bulk_host: str, server_ip: str
) -> None:
    """Demonstrate banking priority using Mininet object."""
    logger.info("Starting banking priority demo")

    # Start bulk transfer to congest link
    logger.info("Starting bulk transfer (P0)")
    generate_traffic_internal(net, TRAFFIC_PROFILES["bulk"], bulk_host, server_ip)

    # Wait for congestion to build
    time.sleep(5)

    # Start banking traffic
    logger.info("Starting banking traffic (P3)")
    generate_traffic_internal(net, TRAFFIC_PROFILES["banking"], banking_host, server_ip)

    logger.info("Banking priority demo running...")
