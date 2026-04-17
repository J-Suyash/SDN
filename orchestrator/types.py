from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional


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


@dataclass
class ClassificationResult:
    priority: str
    confidence: float
    method: str  # 'ml', 'sni', 'port', 'default'
    probabilities: Optional[Dict[str, float]] = None

    def __str__(self) -> str:
        return f"{self.priority} (confidence={self.confidence:.2f}, method={self.method})"
