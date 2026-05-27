"""Fallback port-and-heuristic classifier stub.

Used when the ML model is unavailable.  Delegates port→priority lookup to
the centralised :func:`orchestrator.types.classify_by_port` and applies
simple heuristic rules for the remaining flows (UDP voice, bulk transfers).
"""

from typing import Any, Dict

from orchestrator.types import PriorityClass, Protocol, classify_by_port


def classify_flow(flow_features: Dict[str, Any]) -> str:
    """Classify a flow by port → heuristic chain.

    1. Centralised port→priority lookup (dst_port, then src_port).
    2. Heuristic rules for flows that don't match a known port.
    3. Default to web/office (P1).
    """
    dst_port = flow_features.get("dst_port", 0)
    src_port = flow_features.get("src_port", 0)

    proto = Protocol.from_value(flow_features.get("protocol"))
    bytes_per_sec = flow_features.get("bytes_per_sec", 0.0) or 0.0
    packet_size_avg = flow_features.get("packet_size_avg", 0.0) or 0.0

    # Step 1 — centralised port lookup
    port_priority = classify_by_port(dst_port, src_port)
    if port_priority != "P1":
        return port_priority

    # Step 2 — heuristic fallback
    if proto == Protocol.UDP and 0 < bytes_per_sec < 100_000:
        return PriorityClass.P2_VOICE.label
    if 0 < packet_size_avg < 200:
        return PriorityClass.P2_VOICE.label
    if bytes_per_sec > 5_000_000:
        return PriorityClass.P0_BULK.label

    return PriorityClass.P1_WEB.label
