"""
Collector package for SDN ML Traffic Management.

Uses lazy imports to avoid triggering the Scapy dependency chain
(which requires root) at package import time.
"""

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import to avoid Scapy dependency at package import time."""
    lazy_map: dict[str, str] = {
        "OVSScraper": ".scrape_ovs",
        "collect_stats": ".scrape_ovs",
        "PrometheusScraper": ".scrape_prometheus",
        "collect_faucet_metrics": ".scrape_prometheus",
        "DatasetBuilder": ".build_datasets",
        "FlowRecord": ".build_datasets",
        "LinkRecord": ".build_datasets",
        "label_flow_by_port": ".build_datasets",
        "classify_by_port": ".build_datasets",
        "PacketCapture": ".packet_capture",
    }
    if name in lazy_map:
        module = importlib.import_module(lazy_map[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OVSScraper",
    "collect_stats",
    "PrometheusScraper",
    "collect_faucet_metrics",
    "DatasetBuilder",
    "FlowRecord",
    "LinkRecord",
    "label_flow_by_port",
    "classify_by_port",
    "PacketCapture",
]
