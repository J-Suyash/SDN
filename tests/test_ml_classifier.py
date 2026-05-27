"""Tests for the ML traffic classifier and its fallback chain."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.ml_classifier import MLTrafficClassifier, get_classifier
from orchestrator.types import ClassificationResult


class TestMLClassifierModelLoading(unittest.TestCase):
    """Test model loading logic."""

    def test_init_no_model_graceful(self):
        """Should not crash when model file doesn't exist."""
        clf = MLTrafficClassifier(model_path="/nonexistent/path/model.pkl")
        self.assertIsNotNone(clf)

    def test_default_confidence_threshold(self):
        clf = MLTrafficClassifier(model_path="/nonexistent/path/model.pkl")
        self.assertEqual(clf.confidence_threshold, 0.6)

    def test_custom_confidence_threshold(self):
        clf = MLTrafficClassifier(
            model_path="/nonexistent/path/model.pkl",
            confidence_threshold=0.85,
        )
        self.assertEqual(clf.confidence_threshold, 0.85)

    def test_model_found_in_ml_models(self):
        """The model at ml/models/traffic_classifier.pkl should be found."""
        clf = MLTrafficClassifier()
        self.assertIsNotNone(clf)


class TestMLClassifierFallbackChain(unittest.TestCase):
    """Test the ML -> SNI -> Port fallback chain."""

    def setUp(self):
        self.clf = MLTrafficClassifier(
            model_path="/dev/null/__sdn_test_nonexistent_model.pkl",
        )
        self.clf.model = None  # Force no-model state

    def test_sni_classifies_banking_keywords(self):
        flow = {"sni": "api.razorpay.com"}
        result = self.clf._classify_sni(flow)
        self.assertIsNotNone(result)
        self.assertEqual(result.priority, "P3")
        self.assertEqual(result.method, "sni")

    def test_sni_classifies_voice_keywords(self):
        flow = {"sni": "zoom.us"}
        result = self.clf._classify_sni(flow)
        self.assertIsNotNone(result)
        self.assertEqual(result.priority, "P2")
        self.assertEqual(result.method, "sni")

    def test_sni_classifies_bulk_keywords(self):
        flow = {"sni": "cdn.download.com"}
        result = self.clf._classify_sni(flow)
        self.assertIsNotNone(result)
        self.assertEqual(result.priority, "P0")
        self.assertEqual(result.method, "sni")

    def test_sni_unknown_returns_default_p1(self):
        """Unknown SNI falls back to the SNIClassifier which returns P1."""
        flow = {"sni": "completely.unknown.service"}
        result = self.clf._classify_sni(flow)
        # SNIClassifier returns P1 for unknown domains (not None)
        self.assertIsNotNone(result)
        self.assertEqual(result.priority, "P1")

    def test_sni_empty_returns_none(self):
        flow = {"sni": ""}
        result = self.clf._classify_sni(flow)
        self.assertIsNone(result)

    def test_sni_none_value(self):
        flow = {"sni": "none"}
        result = self.clf._classify_sni(flow)
        self.assertIsNone(result)

    def test_sni_unknown_value(self):
        flow = {"sni": "unknown"}
        result = self.clf._classify_sni(flow)
        self.assertIsNone(result)

    def test_port_classification_443_is_p3(self):
        flow = {"dst_port": 443, "src_port": 54321}
        result = self.clf._classify_port(flow)
        self.assertEqual(result.priority, "P3")
        self.assertEqual(result.method, "port")

    def test_port_classification_5060_is_p2(self):
        flow = {"dst_port": 5060}
        result = self.clf._classify_port(flow)
        self.assertEqual(result.priority, "P2")

    def test_port_classification_80_is_p1(self):
        flow = {"dst_port": 80}
        result = self.clf._classify_port(flow)
        self.assertEqual(result.priority, "P1")

    def test_port_classification_unknown_defaults_p1(self):
        flow = {"dst_port": 9999, "src_port": 10000}
        result = self.clf._classify_port(flow)
        self.assertEqual(result.priority, "P1")
        self.assertEqual(result.method, "default")

    def test_full_classify_falls_through_to_port(self):
        """Without ML model and without SNI, should classify by port."""
        flow = {"dst_port": 443, "src_port": 54321}
        result = self.clf.classify(flow)
        self.assertEqual(result.priority, "P3")
        self.assertEqual(result.method, "port")

    def test_full_classify_uses_sni_over_port(self):
        """SNI should take priority over port fallback."""
        flow = {"dst_port": 80, "src_port": 54321, "sni": "api.razorpay.com"}
        result = self.clf.classify(flow)
        self.assertEqual(result.priority, "P3")
        self.assertEqual(result.method, "sni")

    def test_batch_classify(self):
        flows = [
            {"dst_port": 443, "src_port": 10000},
            {"dst_port": 80, "src_port": 20000},
            {"dst_port": 5002, "src_port": 30000},
        ]
        results = self.clf.classify_batch(flows)
        self.assertEqual(len(results), 3)
        self.assertEqual([r.priority for r in results], ["P3", "P1", "P2"])


class TestMLClassifierStats(unittest.TestCase):
    """Test stats tracking."""

    def setUp(self):
        self.clf = MLTrafficClassifier(
            model_path="/dev/null/__sdn_test_nonexistent_model.pkl",
        )
        self.clf.model = None

    def test_stats_after_classification(self):
        self.clf.classify({"dst_port": 443})
        stats = self.clf.get_stats()
        self.assertEqual(stats["total_classifications"], 1)
        self.assertEqual(stats["port_classifications"], 1)

    def test_stats_ml_not_loaded(self):
        stats = self.clf.get_stats()
        self.assertFalse(stats["model_loaded"])


class TestClassificationResult(unittest.TestCase):
    """Test the ClassificationResult dataclass."""

    def test_str_representation(self):
        result = ClassificationResult(
            priority="P3", confidence=0.95, method="ml",
        )
        s = str(result)
        self.assertIn("P3", s)
        self.assertIn("0.95", s)
        self.assertIn("ml", s)

    def test_with_probabilities(self):
        result = ClassificationResult(
            priority="P3", confidence=0.85, method="ml",
            probabilities={"P3": 0.85, "P2": 0.10, "P1": 0.03, "P0": 0.02},
        )
        self.assertEqual(result.probabilities["P3"], 0.85)


if __name__ == "__main__":
    unittest.main()
