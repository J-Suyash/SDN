from typing import Dict, Any

from orchestrator.types import PriorityClass

PORT_PRIORITY_MAP = {
    443: PriorityClass.P3_BANKING,
    5003: PriorityClass.P3_BANKING,
    5060: PriorityClass.P2_VOICE,
    5061: PriorityClass.P2_VOICE,
    5002: PriorityClass.P2_VOICE,
    80: PriorityClass.P1_WEB,
    8080: PriorityClass.P1_WEB,
    5001: PriorityClass.P1_WEB,
    5000: PriorityClass.P0_BULK,
    21: PriorityClass.P0_BULK,
    22: PriorityClass.P0_BULK,
}


def classify_flow(flow_features: Dict[str, Any]) -> str:
    dst_port = flow_features.get("dst_port", 0)
    src_port = flow_features.get("src_port", 0)
    protocol = flow_features.get("protocol", "tcp")
    if isinstance(protocol, int):
        protocol = "udp" if protocol == 17 else "tcp"
    else:
        protocol = protocol.lower()
    bytes_per_sec = flow_features.get("bytes_per_sec", 0)
    packet_size_avg = flow_features.get("packet_size_avg", 0)

    if dst_port in PORT_PRIORITY_MAP:
        return PORT_PRIORITY_MAP[dst_port].label
    if src_port in PORT_PRIORITY_MAP:
        return PORT_PRIORITY_MAP[src_port].label

    if protocol == "udp" and 0 < bytes_per_sec < 100000:
        return PriorityClass.P2_VOICE.label
    if packet_size_avg > 0 and packet_size_avg < 200:
        return PriorityClass.P2_VOICE.label
    if bytes_per_sec > 5000000:
        return PriorityClass.P0_BULK.label

    return PriorityClass.P1_WEB.label
