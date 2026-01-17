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


def run_iperf_server(host: str, port: int, udp: bool = False) -> subprocess.Popen:
    """
    Start iperf server on a host.

    Args:
        host: Mininet host name (e.g., 'h1')
        port: Port to listen on
        udp: Use UDP mode

    Returns:
        Subprocess handle
    """
    cmd = ["mn", "-c", f"{host} iperf -s -p {port}"]
    if udp:
        cmd[-1] += " -u"
    cmd[-1] += " &"

    logger.info(f"Starting iperf server on {host}:{port} ({'UDP' if udp else 'TCP'})")
    return subprocess.Popen(cmd, shell=True)


def run_iperf_client(
    src_host: str,
    dst_ip: str,
    port: int,
    duration: int,
    bandwidth: Optional[str] = None,
    udp: bool = False,
) -> subprocess.Popen:
    """
    Start iperf client on a host.

    Args:
        src_host: Source Mininet host name
        dst_ip: Destination IP address
        port: Destination port
        duration: Test duration in seconds
        bandwidth: Target bandwidth (e.g., '10M')
        udp: Use UDP mode

    Returns:
        Subprocess handle
    """
    cmd = f"{src_host} iperf -c {dst_ip} -p {port} -t {duration} -i 1"
    if udp:
        cmd += " -u"
    if bandwidth:
        cmd += f" -b {bandwidth}"

    logger.info(f"Starting iperf client: {src_host} -> {dst_ip}:{port}")
    return subprocess.Popen(["mn", "-c", cmd], shell=True)


def generate_traffic(profile: TrafficProfile, src: str, dst_ip: str) -> None:
    """
    Generate traffic based on a profile.

    Args:
        profile: Traffic profile configuration
        src: Source host name
        dst_ip: Destination IP address
    """
    logger.info(f"Generating {profile.name} traffic: {profile.description}")

    is_udp = profile.protocol == "udp"

    # Start server (would be on destination host in real scenario)
    # For MVP, we assume server is already running

    # Start client
    run_iperf_client(
        src_host=src,
        dst_ip=dst_ip,
        port=profile.port,
        duration=profile.duration,
        bandwidth=profile.bandwidth,
        udp=is_udp,
    )


def run_9am_burst_scenario(hosts: List[str], base_ip: str = "10.0.0") -> None:
    """
    Simulate 9AM office login burst.

    Timeline:
    - t=0-60s: Many short bursts (login-like)
    - t=60-180s: Sustained office traffic
    - t=180-240s: Cooldown

    Args:
        hosts: List of host names
        base_ip: Base IP prefix
    """
    logger.info("Starting 9AM burst scenario")

    # Phase 1: Login burst (0-60s)
    logger.info("Phase 1: Login burst")
    for i in range(20):
        host = random.choice(hosts)
        generate_traffic(TRAFFIC_PROFILES["web"], host, f"{base_ip}.1")
        time.sleep(random.uniform(1, 3))

    # Phase 2: Sustained load (60-180s)
    logger.info("Phase 2: Sustained office traffic")
    threads = []
    for host in hosts[1:]:
        t = threading.Thread(
            target=generate_traffic,
            args=(TRAFFIC_PROFILES["web"], host, f"{base_ip}.1"),
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Phase 3: Cooldown (180-240s)
    logger.info("Phase 3: Cooldown")
    time.sleep(60)

    logger.info("9AM burst scenario complete")


def run_banking_priority_demo(
    banking_host: str, bulk_host: str, server_ip: str
) -> None:
    """
    Demonstrate banking priority over bulk traffic.

    1. Start bulk download (P0)
    2. Link becomes congested
    3. Start banking traffic (P3)
    4. Show: Banking gets priority

    Args:
        banking_host: Host for banking traffic
        bulk_host: Host for bulk traffic
        server_ip: Server IP address
    """
    logger.info("Starting banking priority demo")

    # Start bulk transfer to congest link
    logger.info("Starting bulk transfer (P0)")
    bulk_thread = threading.Thread(
        target=generate_traffic, args=(TRAFFIC_PROFILES["bulk"], bulk_host, server_ip)
    )
    bulk_thread.start()

    # Wait for congestion to build
    time.sleep(10)

    # Start banking traffic
    logger.info("Starting banking traffic (P3)")
    generate_traffic(TRAFFIC_PROFILES["banking"], banking_host, server_ip)

    bulk_thread.join()
    logger.info("Banking priority demo complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Traffic Generation Profiles")
    parser.add_argument(
        "--scenario",
        choices=["9am_burst", "banking_priority", "all_profiles"],
        default="all_profiles",
        help="Scenario to run",
    )

    args = parser.parse_args()

    if args.scenario == "9am_burst":
        run_9am_burst_scenario(["h1", "h2", "h3", "h4"])
    elif args.scenario == "banking_priority":
        run_banking_priority_demo("h1", "h2", "10.0.0.3")
    else:
        # Show available profiles
        print("Available traffic profiles:")
        for name, profile in TRAFFIC_PROFILES.items():
            print(f"  {name}: {profile.description}")
