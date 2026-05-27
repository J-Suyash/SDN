"""Tests for the ML congestion predictor."""

import os
import sys
import unittest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct import to avoid scapy chain
import importlib.util
spec = importlib.util.spec_from_file_location(
    'ml_predictor',
    os.path.join(os.getcwd(), 'orchestrator', 'ml_predictor.py')
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MLCongestionPredictor = mod.MLCongestionPredictor


class TestMLCongestionPredictor(unittest.TestCase):
    """Test the ML congestion predictor."""

    def setUp(self):
        self.predictor = MLCongestionPredictor(
            model_path="/dev/null/__sdn_test_nonexistent.pkl",
        )
        self.predictor.model = None  # Force no-model state
        self.predictor.history.clear()  # Start fresh

    def test_init_no_model(self):
        """Should not crash without model file."""
        self.assertIsNotNone(self.predictor)

    def test_update_tracks_link(self):
        self.predictor.update("s1", 3, 0.5)
        self.assertIn("s1:3", self.predictor.history)
        self.assertEqual(len(self.predictor.history["s1:3"]), 1)

    def test_update_increases_history(self):
        self.predictor.update("s1", 3, 0.3)
        self.predictor.update("s1", 3, 0.5)
        self.predictor.update("s1", 3, 0.7)
        self.assertEqual(len(self.predictor.history["s1:3"]), 3)

    def test_update_multiple_links(self):
        self.predictor.update("s1", 3, 0.5)
        self.predictor.update("s2", 4, 0.3)
        self.assertEqual(len(self.predictor.history), 2)
        self.assertIn("s1:3", self.predictor.history)
        self.assertIn("s2:4", self.predictor.history)

    def test_predict_no_history_returns_fallback(self):
        result = self.predictor.predict("s1", 99)
        self.assertEqual(result["method"], "fallback")
        self.assertEqual(result["current_utilization"], 0.0)
        self.assertFalse(result["is_congested"])
        self.assertEqual(result["trend"], "unknown")

    def test_predict_with_history_fallback(self):
        self.predictor.update("s1", 3, 0.3)
        self.predictor.update("s1", 3, 0.4)
        self.predictor.update("s1", 3, 0.5)
        result = self.predictor.predict("s1", 3)
        self.assertEqual(result["method"], "fallback")
        self.assertAlmostEqual(result["current_utilization"], 0.5)

    def test_predict_congestion_threshold(self):
        """Utilization >= 0.80 should flag is_congested."""
        self.predictor.update("s1", 3, 0.85)
        result = self.predictor.predict("s1", 3)
        self.assertTrue(result["is_congested"])

    def test_predict_trend_up_detected(self):
        """Rising utilization should show 'up' trend."""
        for v in [0.3, 0.4, 0.5, 0.6, 0.7]:
            self.predictor.update("s1", 3, v)
        result = self.predictor.predict("s1", 3)
        self.assertEqual(result["trend"], "up")

    def test_predict_trend_down_detected(self):
        """Falling utilization should show 'down' trend."""
        for v in [0.7, 0.6, 0.5, 0.4, 0.3]:
            self.predictor.update("s1", 3, v)
        result = self.predictor.predict("s1", 3)
        self.assertEqual(result["trend"], "down")

    def test_predict_trend_stable(self):
        """Stable utilization around same value."""
        for _ in range(6):
            self.predictor.update("s1", 3, 0.5)
        result = self.predictor.predict("s1", 3)
        self.assertEqual(result["trend"], "stable")

    def test_history_maxlen(self):
        """History should be bounded."""
        for i in range(30):
            self.predictor.update("s1", 3, 0.1 + i * 0.02)
        self.assertLessEqual(
            len(self.predictor.history["s1:3"]),
            self.predictor.max_history,
        )

    def test_get_all_predictions_empty(self):
        preds = self.predictor.get_all_predictions()
        self.assertEqual(preds, {})

    def test_get_all_predictions_with_links(self):
        self.predictor.update("s1", 3, 0.5)
        self.predictor.update("s2", 4, 0.3)
        preds = self.predictor.get_all_predictions()
        self.assertEqual(len(preds), 2)
        self.assertIn("s1:3", preds)
        self.assertIn("s2:4", preds)

    def test_get_stats(self):
        self.predictor.update("s1", 3, 0.5)
        self.predictor.predict("s1", 3)
        stats = self.predictor.get_stats()
        self.assertGreaterEqual(stats["predictions"], 1)
        self.assertGreaterEqual(stats["links_tracked"], 1)

    def test_update_with_timestamp(self):
        ts = datetime(2026, 5, 27, 9, 30, 0)
        self.predictor.update("s1", 3, 0.7, timestamp=ts)
        entry = self.predictor.history["s1:3"][-1]
        self.assertEqual(entry["hour"], 9)
        self.assertEqual(entry["minute"], 30)

    def test_predict_up_trend_fallback_prediction(self):
        """Rising utilization near threshold should predict congestion."""
        for v in [0.50, 0.55, 0.60, 0.62, 0.65]:
            self.predictor.update("s1", 3, v)
        result = self.predictor.predict("s1", 3)
        self.assertEqual(result["trend"], "up")
        self.assertTrue(result["predicted_congestion"])

    def test_fallback_not_triggered_low_utilization(self):
        """Low, stable utilization should not predict congestion."""
        for _ in range(5):
            self.predictor.update("s1", 3, 0.2)
        result = self.predictor.predict("s1", 3)
        self.assertFalse(result["predicted_congestion"])
        self.assertFalse(result["is_congested"])


class TestGetPredictor(unittest.TestCase):
    """Test the singleton accessor."""

    def test_get_predictor_returns_instance(self):
        predictor = mod.get_predictor()
        self.assertIsInstance(predictor, MLCongestionPredictor)

    def test_get_predictor_singleton(self):
        p1 = mod.get_predictor()
        p2 = mod.get_predictor()
        self.assertIs(p1, p2)


if __name__ == "__main__":
    unittest.main()
