import subprocess
import time
import random
import logging
from typing import Optional, List
from dataclasses import dataclass

from orchestrator.types import PriorityClass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrafficProfile:
    name: str
    priority: PriorityClass
    protocol: str  # 'tcp' or 'udp'
    bandwidth: Optional[str] = None
    duration: int = 30
    port: int = 5001
    description: str = ""


TRAFFIC_PROFILES = {
    "banking": TrafficProfile(
        name="banking", priority=PriorityClass.P3_BANKING,
        protocol="tcp", bandwidth="1M", duration=10, port=5003,
        description="Banking/Payment traffic (P3)",
    ),
    "voice": TrafficProfile(
        name="voice", priority=PriorityClass.P2_VOICE,
        protocol="udp", bandwidth="64k", duration=60, port=5002,
        description="Voice/VoIP traffic (P2)",
    ),
    "web": TrafficProfile(
        name="web", priority=PriorityClass.P1_WEB,
        protocol="tcp", bandwidth="2M", duration=30, port=5001,
        description="Web/Office traffic (P1)",
    ),
    "bulk": TrafficProfile(
        name="bulk", priority=PriorityClass.P0_BULK,
        protocol="tcp", bandwidth="10M", duration=120, port=5000,
        description="Bulk download traffic (P0)",
    ),
}


def run_iperf_server(host, port: int, udp: bool = False) -> Optional[int]:
    """Start iperf server on a mininet host. Returns the PID of the background process."""
    cmd = f"iperf -s -p {port}"
    if udp:
        cmd += " -u"
    cmd += " &"
    result = host.cmd(cmd)
    # Extract PID from shell background job output
    pid_cmd = host.cmd("echo $!")
    try:
        return int(pid_cmd.strip())
    except ValueError:
        logger.warning(f"Could not get iperf server PID on {host.name}:{port}")
        return None


def run_iperf_client(host, dst_ip: str, profile: TrafficProfile) -> str:
    """Run iperf client on a mininet host using a traffic profile."""
    cmd = f"iperf -c {dst_ip} -p {profile.port} -t {profile.duration} -i 1"
    if profile.protocol == "udp":
        cmd += " -u"
    if profile.bandwidth:
        cmd += f" -b {profile.bandwidth}"
    cmd += " &"
    return host.cmd(cmd)


def generate_traffic_internal(net, profile: TrafficProfile, src_name: str, dst_ip: str):
    src_host = net.get(src_name)
    logger.info(f"Generating {profile.name} traffic: {profile.description}")
    run_iperf_client(src_host, dst_ip, profile)


def run_9am_burst_scenario(net, hosts: List[str], base_ip: str = "10.0.0") -> None:
    logger.info("Starting 9AM burst scenario")
    for i in range(5):
        host_name = random.choice(hosts)
        generate_traffic_internal(net, TRAFFIC_PROFILES["web"], host_name, f"{base_ip}.3")
        time.sleep(random.uniform(1, 3))
    logger.info("9AM burst scenario complete")


def run_banking_priority_demo(net, banking_host: str, bulk_host: str, server_ip: str) -> None:
    logger.info("Starting banking priority demo")
    generate_traffic_internal(net, TRAFFIC_PROFILES["bulk"], bulk_host, server_ip)
    time.sleep(5)
    generate_traffic_internal(net, TRAFFIC_PROFILES["banking"], banking_host, server_ip)
    logger.info("Banking priority demo running...")
