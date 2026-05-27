"""
ML-based Congestion Predictor

Replaces the threshold-based stub with a trained GradientBoosting model
that predicts link congestion using time-series features (lagged utilization,
rolling statistics, rate of change, and time-of-day patterns).

The model is trained by notebooks/03_train_predictor.py and achieves:
  - F1: ~0.72, ROC-AUC: ~0.94 on synthetic data
  - ~78% recall for congestion events (the metric that matters for proactive rerouting)
"""

import json
import logging
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_joblib = None


class MLCongestionPredictor:
    """
    ML-based congestion predictor using a trained GradientBoosting model.

    Maintains per-link utilization history and extracts time-series features
    (lagged values, rolling statistics, rate of change, time-of-day encoding)
    to predict whether a link will be congested in the next interval.
    """

    DEFAULT_FEATURE_NAMES = [
        "util_lag_1",
        "util_lag_3",
        "util_lag_5",
        "util_lag_10",
        "util_rolling_mean_1",
        "util_rolling_std_1",
        "util_rolling_max_1",
        "util_rolling_mean_3",
        "util_rolling_std_3",
        "util_rolling_max_3",
        "util_rolling_mean_5",
        "util_rolling_std_5",
        "util_rolling_max_5",
        "util_rolling_mean_10",
        "util_rolling_std_10",
        "util_rolling_max_10",
        "util_diff_1",
        "util_diff_3",
        "util_accel",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "is_work_hours",
        "is_morning_peak",
        "is_evening_peak",
    ]

    LOOKBACK_WINDOWS = [1, 3, 5, 10]

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_history: int = 15,
    ):
        self.model = None
        self.model_metadata: Dict[str, Any] = {}
        self.feature_names = list(self.DEFAULT_FEATURE_NAMES)

        # Per-link utilization history: link_id -> deque of dicts
        self.history: Dict[str, deque] = {}
        self.max_history = max(max_history, max(self.LOOKBACK_WINDOWS) + 1)

        self.stats = {
            "predictions": 0,
            "congestion_predicted": 0,
            "links_tracked": 0,
            "errors": 0,
        }

        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the trained congestion predictor model."""
        global _joblib
        if _joblib is None:
            try:
                import joblib
                _joblib = joblib
            except ImportError:
                logger.warning("joblib not available -- ML congestion prediction disabled")
                return False

        search_paths = []
        if model_path:
            search_paths.append(Path(model_path))

        project_root = Path(__file__).parent.parent
        search_paths.extend([
            project_root / "ml" / "models" / "congestion_predictor.pkl",
            project_root / "notebooks" / "ml" / "models" / "congestion_predictor.pkl",
        ])

        model_file = None
        for path in search_paths:
            if path.exists():
                model_file = path
                break

        if model_file is None:
            logger.warning(
                "Congestion predictor model not found. "
                "Run `python notebooks/03_train_predictor.py` first. "
                f"Searched: {[str(p) for p in search_paths[:3]]}"
            )
            return False

        try:
            logger.info(f"Loading congestion predictor from {model_file}")
            self.model = _joblib.load(model_file)

            # Load feature names
            feature_path = model_file.parent / "predictor_features.json"
            if feature_path.exists():
                with open(feature_path) as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")
            else:
                logger.info("Using default feature names (predictor_features.json not found)")

            # Load metadata
            metadata_path = model_file.parent / "predictor_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.model_metadata = json.load(f)

            logger.info(
                f"Congestion predictor loaded "
                f"(model_type={self.model_metadata.get('model_type', 'unknown')})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load congestion predictor model: {e}")
            return False

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None

    def update(
        self, switch: str, port: int, utilization: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update utilization history for a link.

        Args:
            switch: Switch identifier (e.g., 's1')
            port: Port number
            utilization: Current utilization (0.0 to 1.0)
            timestamp: Observation time (defaults to now)
        """
        link_id = f"{switch}:{port}"
        now = timestamp or datetime.now()

        entry = {
            "utilization": float(utilization),
            "hour": now.hour,
            "minute": now.minute,
            "day_of_week": now.weekday(),
            "timestamp": now,
        }

        if link_id not in self.history:
            self.history[link_id] = deque(maxlen=self.max_history)
            self.stats["links_tracked"] += 1

        self.history[link_id].append(entry)

    def predict(self, switch: str, port: int) -> Dict[str, Any]:
        """
        Predict congestion for a link.

        Args:
            switch: Switch identifier
            port: Port number

        Returns:
            Dictionary with:
                - is_congested (bool): current congestion
                - predicted_congestion (bool): predicted in next interval
                - probability (float): prediction confidence (0-1)
                - current_utilization (float): latest utilization
                - trend (str): 'up', 'down', 'stable', or 'unknown'
                - method (str): 'ml' or 'fallback'
        """
        self.stats["predictions"] += 1
        link_id = f"{switch}:{port}"

        current_util = self._get_current_utilization(link_id)

        # If we have enough history for ML prediction
        if (
            link_id in self.history
            and len(self.history[link_id]) >= max(self.LOOKBACK_WINDOWS)
            and self.model is not None
        ):
            try:
                features = self._extract_features(link_id)
                X = np.array([features])

                pred = int(self.model.predict(X)[0])
                proba = float(self.model.predict_proba(X)[0][1])

                if pred:
                    self.stats["congestion_predicted"] += 1

                trend = self._compute_trend(link_id)

                return {
                    "is_congested": bool(current_util >= 0.80),
                    "predicted_congestion": bool(pred),
                    "probability": proba,
                    "current_utilization": current_util,
                    "trend": trend,
                    "method": "ml",
                }
            except Exception as e:
                logger.warning(f"ML prediction failed for {link_id}: {e}")
                self.stats["errors"] += 1

        # Fallback: threshold-based (same as old stub behavior)
        trend = self._compute_trend(link_id) if link_id in self.history else "unknown"
        is_congested = current_util >= 0.80
        predicted = is_congested or (trend == "up" and current_util >= 0.60)

        return {
            "is_congested": is_congested,
            "predicted_congestion": predicted,
            "probability": current_util,
            "current_utilization": current_util,
            "trend": trend,
            "method": "fallback",
        }

    def _get_current_utilization(self, link_id: str) -> float:
        """Get the most recent utilization for a link."""
        if link_id in self.history and self.history[link_id]:
            return self.history[link_id][-1]["utilization"]
        return 0.0

    def _compute_trend(self, link_id: str) -> str:
        """Determine utilization trend from history."""
        if link_id not in self.history or len(self.history[link_id]) < 3:
            return "unknown"

        hist = list(self.history[link_id])
        recent = np.mean([h["utilization"] for h in hist[-3:]])
        older = (
            np.mean([h["utilization"] for h in hist[:-3]])
            if len(hist) > 3
            else recent
        )

        if recent > older + 0.05:
            return "up"
        if recent < older - 0.05:
            return "down"
        return "stable"

    def _extract_features(self, link_id: str) -> List[float]:
        """
        Extract the full feature vector for the ML model.

        Must match the order in predictor_features.json exactly.
        """
        hist = list(self.history[link_id])
        utils = np.array([h["utilization"] for h in hist])
        current = hist[-1]
        n = len(utils)

        feature_values: Dict[str, float] = {}

        # -- Lagged values --
        for lag in self.LOOKBACK_WINDOWS:
            feature_values[f"util_lag_{lag}"] = (
                float(utils[-1 - lag]) if n > lag else float(utils[0])
            )

        # -- Rolling statistics --
        for window in self.LOOKBACK_WINDOWS:
            w = utils[-window:] if n >= window else utils
            feature_values[f"util_rolling_mean_{window}"] = float(np.mean(w))
            feature_values[f"util_rolling_std_{window}"] = (
                float(np.std(w)) if len(w) > 1 else 0.0
            )
            feature_values[f"util_rolling_max_{window}"] = float(np.max(w))

        # -- Differences / rate of change --
        feature_values["util_diff_1"] = (
            float(utils[-1] - utils[-2]) if n > 1 else 0.0
        )
        feature_values["util_diff_3"] = (
            float(utils[-1] - utils[-4]) if n > 3 else 0.0
        )
        feature_values["util_accel"] = (
            float((utils[-1] - utils[-2]) - (utils[-2] - utils[-3]))
            if n > 2
            else 0.0
        )

        # -- Time features --
        hour = current["hour"]
        day = current.get("day_of_week", 0)
        feature_values["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        feature_values["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
        feature_values["day_sin"] = float(np.sin(2 * np.pi * day / 7))
        feature_values["day_cos"] = float(np.cos(2 * np.pi * day / 7))
        feature_values["is_work_hours"] = float(9 <= hour <= 17)
        feature_values["is_morning_peak"] = float(8 <= hour <= 10)
        feature_values["is_evening_peak"] = float(16 <= hour <= 18)

        # Build vector in the exact order of self.feature_names
        vector = [feature_values.get(name, 0.0) for name in self.feature_names]

        # Validate length matches model expectation
        if self.model is not None and hasattr(self.model, "n_features_in_"):
            expected = self.model.n_features_in_
            if len(vector) != expected:
                logger.warning(
                    f"Feature vector length mismatch: got {len(vector)}, "
                    f"model expects {expected}. Padding/truncating."
                )
                if len(vector) < expected:
                    vector.extend([0.0] * (expected - len(vector)))
                else:
                    vector = vector[:expected]

        return vector

    def get_all_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Get predictions for all tracked links."""
        predictions = {}
        for link_id in self.history:
            try:
                switch, port = link_id.split(":")
                predictions[link_id] = self.predict(switch, int(port))
            except (ValueError, Exception):
                continue
        return predictions

    def get_stats(self) -> Dict[str, Any]:
        """Return prediction statistics."""
        total = max(self.stats["predictions"], 1)
        return {
            **self.stats,
            "model_loaded": self.is_loaded(),
            "congestion_rate": f"{self.stats['congestion_predicted'] / total:.1%}",
            "model_accuracy": self.model_metadata.get(
                "test_metrics", {}
            ).get("f1", "N/A"),
            "model_type": self.model_metadata.get("model_type", "N/A"),
        }


# Singleton accessor (matches pattern in ml_classifier.py)
_predictor_instance: Optional["MLCongestionPredictor"] = None


def get_predictor() -> MLCongestionPredictor:
    """Get or create the global congestion predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MLCongestionPredictor()
    return _predictor_instance
