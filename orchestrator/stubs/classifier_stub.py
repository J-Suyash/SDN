"""
Classifier Stub for MVP
Hardcoded classification based on port numbers.
Will be replaced with trained ML model in Phase 4.
"""

from typing import Dict, Any
from enum import Enum


class PriorityClass(Enum):
    """Traffic priority classes"""

    P0_BULK = "P0"  # Bulk/Background - lowest priority
    P1_WEB = "P1"  # Office/Web - best effort
    P2_VOICE = "P2"  # Voice/Video - low jitter
    P3_BANKING = "P3"  # Banking/Payment - highest priority


# Port-based classification rules (MVP stub)
PORT_PRIORITY_MAP = {
    # Banking ports (P3)
    443: PriorityClass.P3_BANKING,  # HTTPS (could be banking)
    5003: PriorityClass.P3_BANKING,  # Custom banking port for demo
    # Voice ports (P2)
    5060: PriorityClass.P2_VOICE,  # SIP
    5061: PriorityClass.P2_VOICE,  # SIP TLS
    5002: PriorityClass.P2_VOICE,  # Custom voice port for demo
    # Web ports (P1)
    80: PriorityClass.P1_WEB,
    8080: PriorityClass.P1_WEB,
    5001: PriorityClass.P1_WEB,  # Default iperf
    # Bulk ports (P0)
    5000: PriorityClass.P0_BULK,  # Custom bulk port for demo
    21: PriorityClass.P0_BULK,  # FTP
    22: PriorityClass.P0_BULK,  # SSH (for file transfer)
}


def classify_flow(flow_features: Dict[str, Any]) -> str:
    """
    Classify a flow based on its features.

    MVP Implementation: Port-based classification
    - Checks destination port against known priority mappings
    - Falls back to protocol-based heuristics

    Args:
        flow_features: Dictionary containing flow information
            - dst_port: Destination port
            - src_port: Source port
            - protocol: 'tcp' or 'udp'
            - bytes_per_sec: Flow rate (optional)
            - packet_size_avg: Average packet size (optional)

    Returns:
        Priority class string: 'P0', 'P1', 'P2', or 'P3'
    """
    dst_port = flow_features.get("dst_port", 0)
    src_port = flow_features.get("src_port", 0)
    protocol = flow_features.get("protocol", "tcp").lower()
    bytes_per_sec = flow_features.get("bytes_per_sec", 0)
    packet_size_avg = flow_features.get("packet_size_avg", 0)

    # Check destination port first
    if dst_port in PORT_PRIORITY_MAP:
        return PORT_PRIORITY_MAP[dst_port].value

    # Check source port (for responses)
    if src_port in PORT_PRIORITY_MAP:
        return PORT_PRIORITY_MAP[src_port].value

    # Protocol-based heuristics
    if protocol == "udp":
        # Low bandwidth UDP is likely voice
        if bytes_per_sec > 0 and bytes_per_sec < 100000:  # < 100 KB/s
            return PriorityClass.P2_VOICE.value
        # High bandwidth UDP could be video
        elif bytes_per_sec > 1000000:  # > 1 MB/s
            return PriorityClass.P2_VOICE.value

    # Small packets with consistent timing suggest voice
    if packet_size_avg > 0 and packet_size_avg < 200:
        return PriorityClass.P2_VOICE.value

    # Large flows are likely bulk
    if bytes_per_sec > 5000000:  # > 5 MB/s
        return PriorityClass.P0_BULK.value

    # Default to web/office
    return PriorityClass.P1_WEB.value


def get_priority_description(priority: str) -> str:
    """Get human-readable description of priority class"""
    descriptions = {
        "P0": "Bulk/Background (lowest priority)",
        "P1": "Web/Office (best effort)",
        "P2": "Voice/Video (low jitter)",
        "P3": "Banking/Payment (highest priority)",
    }
    return descriptions.get(priority, "Unknown")


if __name__ == "__main__":
    # Test the classifier
    test_flows = [
        {"dst_port": 5003, "protocol": "tcp"},  # Banking
        {"dst_port": 5002, "protocol": "udp", "bytes_per_sec": 50000},  # Voice
        {"dst_port": 80, "protocol": "tcp"},  # Web
        {"dst_port": 5000, "protocol": "tcp", "bytes_per_sec": 10000000},  # Bulk
        {"dst_port": 12345, "protocol": "udp", "bytes_per_sec": 30000},  # Unknown UDP
    ]

    print("Classifier Stub Test:")
    print("-" * 50)
    for flow in test_flows:
        priority = classify_flow(flow)
        desc = get_priority_description(priority)
        print(f"Flow {flow} -> {priority}: {desc}")
