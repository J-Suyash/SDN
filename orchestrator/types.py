from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Set


class PriorityClass(Enum):
    P0_BULK = ("P0", 3, "Bulk/Background")
    P1_WEB = ("P1", 2, "Web/Office")
    P2_VOICE = ("P2", 1, "Voice/Video")
    P3_BANKING = ("P3", 0, "Banking/Payment")

    def __init__(self, label: str, queue_id: int, description: str):
        self.label = label
        self.queue_id = queue_id
        self.description = description

    @classmethod
    def from_label(cls, label: str) -> "PriorityClass":
        for member in cls:
            if member.label == label:
                return member
        return cls.P1_WEB


PRIORITY_QUEUES = {
    "P3": 0,
    "P2": 1,
    "P1": 2,
    "P0": 3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Port-to-Priority Mapping  —  Single Source of Truth
# ═══════════════════════════════════════════════════════════════════════════════
# All modules MUST import from here.  Do NOT duplicate these maps elsewhere.

PORT_PRIORITY_MAP: Dict[int, str] = {
    # P3 — Banking / Payment
    443: "P3", 5003: "P3", 8443: "P3",
    # P2 — Voice / Video
    5060: "P2", 5061: "P2", 5002: "P2", 3478: "P2", 3479: "P2",
    # P0 — Bulk / Background
    20: "P0", 21: "P0", 22: "P0", 5000: "P0",
    # P1 — Web / Office
    80: "P1", 8080: "P1", 5001: "P1",
}

P3_PORTS: Set[int] = {p for p, v in PORT_PRIORITY_MAP.items() if v == "P3"}
P2_PORTS: Set[int] = {p for p, v in PORT_PRIORITY_MAP.items() if v == "P2"}
P1_PORTS: Set[int] = {p for p, v in PORT_PRIORITY_MAP.items() if v == "P1"}
P0_PORTS: Set[int] = {p for p, v in PORT_PRIORITY_MAP.items() if v == "P0"}


def classify_by_port(dst_port: int, src_port: int = 0) -> str:
    """Classify a flow's priority class based on port numbers.

    Checks *dst_port* first, then *src_port*.  Returns ``"P1"`` when
    neither port is a known service port.
    """
    if dst_port in PORT_PRIORITY_MAP:
        return PORT_PRIORITY_MAP[dst_port]
    if src_port in PORT_PRIORITY_MAP:
        return PORT_PRIORITY_MAP[src_port]
    return "P1"


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol Enum
# ═══════════════════════════════════════════════════════════════════════════════

class Protocol(Enum):
    TCP = 6
    UDP = 17

    @classmethod
    def from_value(cls, value: object) -> "Protocol":
        """Normalise an ``int`` (6/17), ``str`` (``"tcp"`` / ``"udp"`` / ``"6"`` / ``"17"``),
        or ``None`` to a :class:`Protocol`.
        """
        if value is None:
            return cls.TCP
        if isinstance(value, int):
            for p in cls:
                if p.value == value:
                    return p
            return cls.TCP
        s = str(value).lower().strip()
        if s in ("udp", "17"):
            return cls.UDP
        return cls.TCP

    @property
    def name_lower(self) -> str:
        """Lowercase name for OVS match strings (``"tcp"`` / ``"udp"``)."""
        return "tcp" if self == Protocol.TCP else "udp"


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Constants  —  Single Source of Truth
# ═══════════════════════════════════════════════════════════════════════════════

# Fraction of link capacity above which a link is considered *currently* congested.
CONGESTION_THRESHOLD: float = 0.80

# Fraction of link capacity above which the orchestrator *predicts* congestion.
PREDICTION_THRESHOLD: float = 0.70

# Default control-loop interval in seconds.
POLL_INTERVAL: int = 10

# Default link capacity (bps) used when no per-link capacity is configured.
DEFAULT_LINK_CAPACITY_BPS: int = 10_000_000  # 10 Mbps

# Seconds before an idle flow is pruned from the capture cache.
CAPTURE_IDLE_TIMEOUT: int = 30

# Prometheus exporter port.
ORCHESTRATOR_METRICS_PORT: int = 8000

# Default Faucet / Prometheus URLs.
PROMETHEUS_URL: str = "http://localhost:9090"

# Default data directory for ML datasets.
DATA_DIR: str = "/app/data/processed"

# Comma-separated Mininet interfaces to capture on.
CAPTURE_INTERFACES: str = "s1-eth1,s1-eth2"


@dataclass
class ClassificationResult:
    priority: str
    confidence: float
    method: str  # 'ml', 'sni', 'port', 'default'
    probabilities: Optional[Dict[str, float]] = None

    def __str__(self) -> str:
        return f"{self.priority} (confidence={self.confidence:.2f}, method={self.method})"
