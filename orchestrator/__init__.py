"""
Orchestrator package for SDN ML Traffic Management.

Key classes are available via lazy imports to avoid triggering
the Scapy dependency chain (which requires root) at import time.
"""

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import to avoid Scapy dependency at package import time."""
    lazy_map = {
        "Orchestrator": ".orchestrator",
        "PolicyEngine": ".policy_engine",
        "Action": ".policy_engine",
        "PolicyDecision": ".policy_engine",
    }
    if name in lazy_map:
        module = importlib.import_module(lazy_map[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Orchestrator", "PolicyEngine", "Action", "PolicyDecision"]
