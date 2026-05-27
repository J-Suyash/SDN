"""
Congestion Predictor — ML-powered with threshold fallback

Uses the trained MLCongestionPredictor model when available.
Falls back to threshold-based logic when the model is not loaded.
Maintains the same interface (update_link, predict, get_all_predictions)
for backward compatibility with the orchestrator.
"""

import logging
import time
import os
from typing import Dict, Any, List, Optional

from orchestrator.ml_predictor import MLCongestionPredictor, get_predictor

logger = logging.getLogger(__name__)

# Thresholds for fallback mode
CONGESTION_THRESHOLD = 0.80
PREDICTION_THRESHOLD = 0.60

# Global predictor instance
_predictor: Optional[MLCongestionPredictor] = None


def _get_ml_predictor() -> Optional[MLCongestionPredictor]:
    """Get or create the ML predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = get_predictor()
    return _predictor if _predictor.is_loaded() else None


def update_link(
    switch: str,
    port: int,
    bytes_count: int,
    capacity_bps: int = 10_000_000,
    interval_sec: int = 10,
) -> None:
    """
    Update link statistics and feed into the ML predictor.

    Args:
        switch: Switch identifier
        port: Port number
        bytes_count: Current byte counter value
        capacity_bps: Link capacity in bits per second
        interval_sec: Collection interval in seconds
    """
    ml = _get_ml_predictor()
    if ml is not None:
        # Need utilization - compute from bytes delta
        # We store the last bytes value per link to compute delta
        key = f"{switch}:{port}"
        _update_bytes_cache(key, bytes_count, capacity_bps, interval_sec)
        return

    # Legacy threshold-based update (for fallback)
    _update_threshold_fallback(switch, port, bytes_count, capacity_bps)


# Byte cache for computing deltas
_bytes_cache: Dict[str, Dict[str, Any]] = {}


def _update_bytes_cache(
    key: str,
    bytes_count: int,
    capacity_bps: int,
    interval_sec: int,
) -> None:
    """Compute utilization from byte delta and feed into ML predictor."""
    ml = _get_ml_predictor()
    if ml is None:
        return

    now = time.time()
    switch, port_str = key.split(":")
    port = int(port_str)

    if key in _bytes_cache:
        prev = _bytes_cache[key]
        time_delta = now - prev["time"]
        if time_delta > 0:
            bytes_delta = bytes_count - prev["bytes"]
            bits_delta = bytes_delta * 8
            utilization = bits_delta / (capacity_bps * time_delta)
            utilization = min(1.0, max(0.0, utilization))
            ml.update(switch, port, utilization)

    _bytes_cache[key] = {"bytes": bytes_count, "time": now}


def predict(switch: str, port: int) -> Dict[str, Any]:
    """
    Predict congestion for a link.

    Uses ML model when available, falls back to threshold-based logic.

    Returns:
        Dictionary with prediction results matching the original stub format:
        - is_congested: bool
        - predicted_congestion: bool
        - current_utilization: float
        - trend: str ('up', 'down', 'stable', 'unknown')
    """
    ml = _get_ml_predictor()
    if ml is not None:
        result = ml.predict(switch, port)
        return {
            "is_congested": result["is_congested"],
            "predicted_congestion": result["predicted_congestion"],
            "current_utilization": result["current_utilization"],
            "trend": result["trend"],
        }

    return _fallback_predict(switch, port)


def _update_threshold_fallback(
    switch: str, port: int, bytes_count: int, capacity_bps: int
) -> None:
    """Legacy threshold-based stub logic."""
    key = f"{switch}:{port}"
    current_time = time.time()

    if key in _bytes_cache:
        prev = _bytes_cache[key]
        time_delta = current_time - prev["time"]
        if time_delta > 0:
            bytes_delta = bytes_count - prev["bytes"]
            bits_delta = bytes_delta * 8
            utilization = bits_delta / (capacity_bps * time_delta)
            utilization = min(1.0, max(0.0, utilization))

            if key not in _link_histories:
                _link_histories[key] = []
            _link_histories[key].append(utilization)
            if len(_link_histories[key]) > 10:
                _link_histories[key].pop(0)

    _bytes_cache[key] = {"bytes": bytes_count, "time": current_time}


_link_histories: Dict[str, List[float]] = {}


def _fallback_predict(switch: str, port: int) -> Dict[str, Any]:
    """Threshold-based fallback prediction."""
    key = f"{switch}:{port}"
    history = _link_histories.get(key, [])

    if not history:
        return {
            "is_congested": False,
            "predicted_congestion": False,
            "current_utilization": 0.0,
            "trend": "unknown",
        }

    current_util = history[-1]

    trend = "stable"
    if len(history) >= 3:
        recent_avg = sum(history[-3:]) / 3
        older_avg = (
            sum(history[:-3]) / max(1, len(history) - 3)
            if len(history) > 3
            else recent_avg
        )
        if recent_avg > older_avg + 0.1:
            trend = "up"
        elif recent_avg < older_avg - 0.1:
            trend = "down"

    is_congested = current_util > CONGESTION_THRESHOLD
    predicted_congestion = (
        (trend == "up" and current_util > PREDICTION_THRESHOLD) or is_congested
    )

    return {
        "is_congested": is_congested,
        "predicted_congestion": predicted_congestion,
        "current_utilization": current_util,
        "trend": trend,
    }


def get_all_predictions() -> Dict[str, Dict[str, Any]]:
    """Get congestion predictions for all known links."""
    ml = _get_ml_predictor()
    if ml is not None:
        raw = ml.get_all_predictions()
        result = {}
        for link_id, pred in raw.items():
            result[link_id] = {
                "is_congested": pred["is_congested"],
                "predicted_congestion": pred["predicted_congestion"],
                "current_utilization": pred["current_utilization"],
                "trend": pred["trend"],
            }
        return result

    predictions = {}
    for key in _link_histories:
        switch, port = key.split(":")
        predictions[key] = _fallback_predict(switch, int(port))
    return predictions


def get_stats() -> Dict[str, Any]:
    """Get predictor statistics."""
    ml = _get_ml_predictor()
    if ml is not None:
        return ml.get_stats()
    return {
        "predictions": 0,
        "links_tracked": len(_link_histories),
        "model_loaded": False,
        "model_type": "threshold_stub",
    }