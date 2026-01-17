"""
Predictor Stub for MVP
Hardcoded congestion prediction based on utilization threshold.
Will be replaced with trained ML model in Phase 4.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time


@dataclass
class LinkState:
    """State of a network link"""

    switch: str
    port: int
    capacity_bps: int  # bits per second
    current_bytes: int
    last_update: float
    history: List[float]  # utilization history


# Thresholds for congestion prediction (MVP)
CONGESTION_THRESHOLD = 0.7  # 70% utilization
PREDICTION_THRESHOLD = 0.6  # Predict congestion if trending toward 70%


class PredictorStub:
    """
    MVP Congestion Predictor

    Uses simple threshold-based rules:
    - If current utilization > 70%: congested
    - If trending upward and > 60%: predict congestion
    """

    def __init__(self):
        self.link_states: Dict[str, LinkState] = {}

    def update_link_stats(
        self,
        switch: str,
        port: int,
        bytes_count: int,
        capacity_bps: int = 10_000_000,  # 10 Mbps default
    ) -> None:
        """
        Update link statistics.

        Args:
            switch: Switch identifier
            port: Port number
            bytes_count: Current byte counter
            capacity_bps: Link capacity in bits per second
        """
        key = f"{switch}:{port}"
        current_time = time.time()

        if key in self.link_states:
            state = self.link_states[key]

            # Calculate utilization for this interval
            time_delta = current_time - state.last_update
            if time_delta > 0:
                bytes_delta = bytes_count - state.current_bytes
                bits_delta = bytes_delta * 8
                utilization = bits_delta / (capacity_bps * time_delta)
                utilization = min(1.0, max(0.0, utilization))  # Clamp to [0, 1]

                # Update history (keep last 10 samples)
                state.history.append(utilization)
                if len(state.history) > 10:
                    state.history.pop(0)

            state.current_bytes = bytes_count
            state.last_update = current_time
        else:
            # Initialize new link state
            self.link_states[key] = LinkState(
                switch=switch,
                port=port,
                capacity_bps=capacity_bps,
                current_bytes=bytes_count,
                last_update=current_time,
                history=[],
            )

    def predict_congestion(self, switch: str, port: int) -> Dict[str, Any]:
        """
        Predict if a link will be congested.

        MVP Implementation:
        - Current utilization > 70%: congested now
        - Trending up and > 60%: predict congestion

        Args:
            switch: Switch identifier
            port: Port number

        Returns:
            Dictionary with prediction results:
            - is_congested: bool
            - predicted_congestion: bool
            - current_utilization: float
            - trend: str ('up', 'down', 'stable')
        """
        key = f"{switch}:{port}"

        if key not in self.link_states:
            return {
                "is_congested": False,
                "predicted_congestion": False,
                "current_utilization": 0.0,
                "trend": "unknown",
            }

        state = self.link_states[key]

        if not state.history:
            return {
                "is_congested": False,
                "predicted_congestion": False,
                "current_utilization": 0.0,
                "trend": "unknown",
            }

        current_util = state.history[-1]

        # Determine trend
        trend = "stable"
        if len(state.history) >= 3:
            recent_avg = sum(state.history[-3:]) / 3
            older_avg = (
                sum(state.history[:-3]) / max(1, len(state.history) - 3)
                if len(state.history) > 3
                else recent_avg
            )

            if recent_avg > older_avg + 0.1:
                trend = "up"
            elif recent_avg < older_avg - 0.1:
                trend = "down"

        # Current congestion check
        is_congested = current_util > CONGESTION_THRESHOLD

        # Predict future congestion
        predicted_congestion = False
        if trend == "up" and current_util > PREDICTION_THRESHOLD:
            predicted_congestion = True
        elif current_util > CONGESTION_THRESHOLD:
            predicted_congestion = True

        return {
            "is_congested": is_congested,
            "predicted_congestion": predicted_congestion,
            "current_utilization": current_util,
            "trend": trend,
        }

    def get_all_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Get congestion predictions for all known links"""
        predictions = {}
        for key in self.link_states:
            switch, port = key.split(":")
            predictions[key] = self.predict_congestion(switch, int(port))
        return predictions


# Global predictor instance
_predictor = PredictorStub()


def update_link(
    switch: str, port: int, bytes_count: int, capacity_bps: int = 10_000_000
) -> None:
    """Update link statistics"""
    _predictor.update_link_stats(switch, port, bytes_count, capacity_bps)


def predict(switch: str, port: int) -> Dict[str, Any]:
    """Predict congestion for a link"""
    return _predictor.predict_congestion(switch, port)


def get_all_predictions() -> Dict[str, Dict[str, Any]]:
    """Get all link predictions"""
    return _predictor.get_all_predictions()


if __name__ == "__main__":
    # Test the predictor
    import random

    print("Predictor Stub Test:")
    print("-" * 50)

    # Simulate increasing utilization
    predictor = PredictorStub()

    bytes_count = 0
    for i in range(15):
        # Simulate traffic increasing over time
        bytes_count += random.randint(500000, 1500000)  # 0.5-1.5 MB per second
        predictor.update_link_stats("s1", 1, bytes_count, 10_000_000)

        result = predictor.predict_congestion("s1", 1)
        print(
            f"Step {i + 1}: util={result['current_utilization']:.2%}, "
            f"trend={result['trend']}, "
            f"congested={result['is_congested']}, "
            f"predicted={result['predicted_congestion']}"
        )

        time.sleep(0.1)  # Small delay for timestamp differences
