"""
ML-Based Traffic Classifier for SDN Traffic Management.

This module provides production-grade traffic classification using a trained
scikit-learn RandomForest model. It integrates with the policy engine to
classify network flows into priority classes (P0-P3).

Architecture:
- Primary: ML model trained on real traffic captures
- Fallback: SNI-based classification (if SNI available)
- Fallback: Port-based heuristics (last resort)

Usage:
    classifier = MLTrafficClassifier()
    priority, confidence = classifier.classify(flow_features)
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Lazy import for joblib to handle missing dependency gracefully
_joblib = None
_model_loaded = False

logger = logging.getLogger("ml_classifier")


class PriorityClass(Enum):
    """Traffic priority classes with QoS queue mapping."""

    P0_BULK = "P0"  # Bulk/Background - lowest priority (Queue 3)
    P1_WEB = "P1"  # Office/Web - best effort (Queue 2)
    P2_VOICE = "P2"  # Voice/Video - low jitter (Queue 1)
    P3_BANKING = "P3"  # Banking/Payment - highest priority (Queue 0)


@dataclass
class ClassificationResult:
    """Result of flow classification with metadata."""

    priority: str
    confidence: float
    method: str  # 'ml', 'sni', 'port', 'default'
    probabilities: Optional[Dict[str, float]] = None

    def __str__(self) -> str:
        return (
            f"{self.priority} (confidence={self.confidence:.2f}, method={self.method})"
        )


class MLTrafficClassifier:
    """
    Production ML-based traffic classifier.

    Uses a trained RandomForest model to classify network flows into
    priority classes. Provides confidence scores and fallback mechanisms.

    Attributes:
        model: Trained sklearn Pipeline (scaler + classifier)
        label_encoder: LabelEncoder for class name mapping
        feature_names: List of expected feature column names
        confidence_threshold: Minimum confidence for ML prediction

    Example:
        >>> classifier = MLTrafficClassifier()
        >>> flow = {
        ...     'packet_count': 150,
        ...     'byte_count': 45000,
        ...     'duration_sec': 2.5,
        ...     'bytes_per_packet': 300,
        ...     'packets_per_sec': 60,
        ...     'bytes_per_sec': 18000,
        ...     'pkt_len_min': 64,
        ...     'pkt_len_max': 1200,
        ...     'pkt_len_mean': 300,
        ...     'pkt_len_std': 150,
        ...     'iat_mean': 0.017,
        ...     'iat_std': 0.008
        ... }
        >>> result = classifier.classify(flow)
        >>> print(result.priority, result.confidence)
    """

    # Default feature names matching the trained model
    DEFAULT_FEATURE_NAMES = [
        "packet_count",
        "byte_count",
        "duration_sec",
        "bytes_per_packet",
        "packets_per_sec",
        "bytes_per_sec",
        "pkt_len_min",
        "pkt_len_max",
        "pkt_len_mean",
        "pkt_len_std",
        "iat_mean",
        "iat_std",
    ]

    # Port-based fallback classification
    PORT_PRIORITY_MAP = {
        # Banking ports (P3)
        443: "P3",
        5003: "P3",
        8443: "P3",
        # Voice ports (P2)
        5060: "P2",
        5061: "P2",
        5002: "P2",
        3478: "P2",
        3479: "P2",
        # Bulk ports (P0)
        20: "P0",
        21: "P0",
        22: "P0",
        5000: "P0",
        # Web ports (P1)
        80: "P1",
        8080: "P1",
        5001: "P1",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
        enable_fallback: bool = True,
    ):
        """
        Initialize the ML classifier.

        Args:
            model_path: Path to the trained model (.pkl file).
                        If None, searches default locations.
            confidence_threshold: Minimum confidence to trust ML prediction.
                                 Below this, fallback methods are used.
            enable_fallback: Whether to use SNI/port fallback when ML fails.
        """
        self.model = None
        self.label_encoder = None
        self.feature_names = self.DEFAULT_FEATURE_NAMES.copy()
        self.confidence_threshold = confidence_threshold
        self.enable_fallback = enable_fallback
        self.model_metadata = {}

        # Statistics
        self.stats = {
            "total_classifications": 0,
            "ml_classifications": 0,
            "sni_classifications": 0,
            "port_classifications": 0,
            "default_classifications": 0,
            "low_confidence_fallbacks": 0,
        }

        # Try to load the model
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the trained model and associated artifacts.

        Args:
            model_path: Path to model file, or None to search defaults.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        global _joblib, _model_loaded

        # Import joblib lazily
        if _joblib is None:
            try:
                import joblib

                _joblib = joblib
            except ImportError:
                logger.warning("joblib not available. ML classification disabled.")
                return False

        # Search paths for model
        search_paths = []
        if model_path:
            search_paths.append(Path(model_path))

        # Default search locations
        project_root = Path(__file__).parent.parent
        search_paths.extend(
            [
                project_root / "notebooks" / "ml" / "models" / "traffic_classifier.pkl",
                project_root / "ml" / "models" / "traffic_classifier.pkl",
                Path("ml/models/traffic_classifier.pkl"),
                Path("notebooks/ml/models/traffic_classifier.pkl"),
            ]
        )

        # Find and load model
        model_file = None
        for path in search_paths:
            if path.exists():
                model_file = path
                break

        if model_file is None:
            logger.warning(f"Model file not found. Searched: {search_paths[:2]}")
            return False

        try:
            logger.info(f"Loading model from {model_file}")
            self.model = _joblib.load(model_file)

            # Load label encoder
            encoder_path = model_file.parent / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = _joblib.load(encoder_path)
                logger.info(
                    f"Loaded label encoder: {list(self.label_encoder.classes_)}"
                )

            # Load feature names
            feature_path = model_file.parent / "feature_names.json"
            if feature_path.exists():
                with open(feature_path) as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")

            # Load metadata
            metadata_path = model_file.parent / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.model_metadata = json.load(f)
                logger.info(
                    f"Model accuracy: {self.model_metadata.get('test_metrics', {}).get('accuracy', 'N/A')}"
                )

            _model_loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _extract_features(self, flow: Dict[str, Any]) -> np.ndarray:
        """
        Extract feature vector from flow dictionary.

        Args:
            flow: Flow dictionary with traffic features.

        Returns:
            numpy array of features in expected order.
        """
        features = []
        for name in self.feature_names:
            value = flow.get(name, 0)
            # Handle None values
            if value is None:
                value = 0
            features.append(float(value))

        return np.array([features])

    def _classify_ml(self, flow: Dict[str, Any]) -> Optional[ClassificationResult]:
        """
        Classify using ML model.

        Returns:
            ClassificationResult or None if ML unavailable.
        """
        if self.model is None:
            return None

        try:
            X = self._extract_features(flow)

            # Get prediction and probabilities
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]

            # Map prediction to label
            if self.label_encoder is not None:
                # pred is already the class index
                priority = self.label_encoder.inverse_transform([pred])[0]
                class_probas = {
                    self.label_encoder.classes_[i]: float(p)
                    for i, p in enumerate(proba)
                }
            else:
                # Assume pred is the label directly
                priority = str(pred)
                class_probas = {f"class_{i}": float(p) for i, p in enumerate(proba)}

            confidence = float(max(proba))

            return ClassificationResult(
                priority=priority,
                confidence=confidence,
                method="ml",
                probabilities=class_probas,
            )

        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return None

    def _classify_sni(self, flow: Dict[str, Any]) -> Optional[ClassificationResult]:
        """
        Classify based on SNI domain.

        Returns:
            ClassificationResult or None if no SNI available.
        """
        sni = flow.get("sni", "") or flow.get("sni_domain", "")
        if not sni or sni.lower() in ("", "unknown", "none"):
            return None

        # Try to use SNI classifier if available
        try:
            from orchestrator.sni_classifier import SNIClassifier

            classifier = SNIClassifier()
            priority = classifier.classify(sni)

            # SNI gives high confidence if matched
            return ClassificationResult(
                priority=priority,
                confidence=0.9 if priority != "P1" else 0.5,  # P1 is default
                method="sni",
            )
        except ImportError:
            pass

        # Fallback: basic domain matching
        sni_lower = sni.lower()

        # Banking domains
        if any(
            kw in sni_lower
            for kw in ["bank", "payment", "pay", "razorpay", "stripe", "paytm"]
        ):
            return ClassificationResult(priority="P3", confidence=0.85, method="sni")

        # Voice domains
        if any(
            kw in sni_lower for kw in ["meet", "zoom", "teams", "webrtc", "voip", "sip"]
        ):
            return ClassificationResult(priority="P2", confidence=0.85, method="sni")

        # Bulk domains
        if any(
            kw in sni_lower
            for kw in ["cdn", "download", "update", "release", "iso", "torrent"]
        ):
            return ClassificationResult(priority="P0", confidence=0.75, method="sni")

        return None

    def _classify_port(self, flow: Dict[str, Any]) -> ClassificationResult:
        """
        Classify based on port numbers.

        Returns:
            ClassificationResult (always returns a result).
        """
        dst_port = flow.get("dst_port", 0)
        src_port = flow.get("src_port", 0)

        # Check destination port first
        if dst_port in self.PORT_PRIORITY_MAP:
            return ClassificationResult(
                priority=self.PORT_PRIORITY_MAP[dst_port], confidence=0.7, method="port"
            )

        # Check source port
        if src_port in self.PORT_PRIORITY_MAP:
            return ClassificationResult(
                priority=self.PORT_PRIORITY_MAP[src_port], confidence=0.6, method="port"
            )

        # Default to web traffic
        return ClassificationResult(priority="P1", confidence=0.3, method="default")

    def classify(self, flow: Dict[str, Any]) -> ClassificationResult:
        """
        Classify a network flow into a priority class.

        Uses a cascading approach:
        1. ML model (if available and confident)
        2. SNI-based classification (if SNI available)
        3. Port-based classification (fallback)

        Args:
            flow: Dictionary containing flow features.
                  Required for ML: packet_count, byte_count, duration_sec, etc.
                  Optional: sni, dst_port, src_port

        Returns:
            ClassificationResult with priority, confidence, and method used.
        """
        self.stats["total_classifications"] += 1

        # Try ML classification first
        ml_result = self._classify_ml(flow)
        if ml_result is not None:
            if ml_result.confidence >= self.confidence_threshold:
                self.stats["ml_classifications"] += 1
                return ml_result
            else:
                self.stats["low_confidence_fallbacks"] += 1
                logger.debug(
                    f"Low ML confidence ({ml_result.confidence:.2f}), using fallback"
                )

        # Fallback to SNI if enabled
        if self.enable_fallback:
            sni_result = self._classify_sni(flow)
            if sni_result is not None:
                self.stats["sni_classifications"] += 1
                return sni_result

        # Port-based fallback
        port_result = self._classify_port(flow)
        if port_result.method == "port":
            self.stats["port_classifications"] += 1
        else:
            self.stats["default_classifications"] += 1

        return port_result

    def classify_batch(self, flows: List[Dict[str, Any]]) -> List[ClassificationResult]:
        """
        Classify multiple flows efficiently.

        Args:
            flows: List of flow dictionaries.

        Returns:
            List of ClassificationResult objects.
        """
        return [self.classify(flow) for flow in flows]

    def get_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        total = self.stats["total_classifications"]
        if total == 0:
            total = 1  # Avoid division by zero

        return {
            **self.stats,
            "ml_rate": f"{self.stats['ml_classifications'] / total:.1%}",
            "sni_rate": f"{self.stats['sni_classifications'] / total:.1%}",
            "port_rate": f"{self.stats['port_classifications'] / total:.1%}",
            "model_loaded": self.model is not None,
            "model_accuracy": self.model_metadata.get("test_metrics", {}).get(
                "accuracy", "N/A"
            ),
        }

    def is_model_loaded(self) -> bool:
        """Check if ML model is available."""
        return self.model is not None


# Convenience function for simple usage
def classify_flow(flow: Dict[str, Any]) -> str:
    """
    Simple function to classify a flow.

    Args:
        flow: Flow dictionary with features.

    Returns:
        Priority string: 'P0', 'P1', 'P2', or 'P3'
    """
    classifier = MLTrafficClassifier()
    result = classifier.classify(flow)
    return result.priority


def get_priority_description(priority: str) -> str:
    """Get human-readable description of priority class."""
    descriptions = {
        "P0": "Bulk/Background (lowest priority)",
        "P1": "Web/Office (best effort)",
        "P2": "Voice/Video (low jitter)",
        "P3": "Banking/Payment (highest priority)",
    }
    return descriptions.get(priority, "Unknown")


# Module-level classifier instance (lazy initialization)
_classifier_instance: Optional[MLTrafficClassifier] = None


def get_classifier() -> MLTrafficClassifier:
    """Get or create the singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = MLTrafficClassifier()
    return _classifier_instance


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    print("=" * 70)
    print("ML Traffic Classifier - Integration Test")
    print("=" * 70)

    classifier = MLTrafficClassifier()

    if not classifier.is_model_loaded():
        print("\n⚠ WARNING: ML model not loaded. Using fallback classification.")
        print("  Run notebooks/02_train_classifier.py first to train the model.")
    else:
        print(f"\n✓ Model loaded successfully")
        print(
            f"  Accuracy: {classifier.model_metadata.get('test_metrics', {}).get('accuracy', 'N/A')}"
        )

    # Test flows representing different traffic types
    test_flows = [
        # Banking-like flow (small packets, bursty)
        {
            "flow_id": "banking-test",
            "packet_count": 50,
            "byte_count": 15000,
            "duration_sec": 2.0,
            "bytes_per_packet": 300,
            "packets_per_sec": 25,
            "bytes_per_sec": 7500,
            "pkt_len_min": 64,
            "pkt_len_max": 500,
            "pkt_len_mean": 300,
            "pkt_len_std": 100,
            "iat_mean": 0.04,
            "iat_std": 0.02,
            "sni": "netbanking.hdfcbank.com",
            "dst_port": 443,
        },
        # Voice-like flow (consistent small packets)
        {
            "flow_id": "voice-test",
            "packet_count": 500,
            "byte_count": 100000,
            "duration_sec": 30.0,
            "bytes_per_packet": 200,
            "packets_per_sec": 16.7,
            "bytes_per_sec": 3333,
            "pkt_len_min": 160,
            "pkt_len_max": 240,
            "pkt_len_mean": 200,
            "pkt_len_std": 20,
            "iat_mean": 0.02,
            "iat_std": 0.005,
            "sni": "meet.google.com",
            "dst_port": 5060,
        },
        # Bulk download (large packets, sustained)
        {
            "flow_id": "bulk-test",
            "packet_count": 2000,
            "byte_count": 3000000,
            "duration_sec": 30.0,
            "bytes_per_packet": 1500,
            "packets_per_sec": 66.7,
            "bytes_per_sec": 100000,
            "pkt_len_min": 1200,
            "pkt_len_max": 1500,
            "pkt_len_mean": 1450,
            "pkt_len_std": 100,
            "iat_mean": 0.0001,
            "iat_std": 0.00005,
            "sni": "releases.ubuntu.com",
            "dst_port": 5000,
        },
        # Web browsing (mixed packets)
        {
            "flow_id": "web-test",
            "packet_count": 100,
            "byte_count": 80000,
            "duration_sec": 5.0,
            "bytes_per_packet": 800,
            "packets_per_sec": 20,
            "bytes_per_sec": 16000,
            "pkt_len_min": 64,
            "pkt_len_max": 1500,
            "pkt_len_mean": 800,
            "pkt_len_std": 400,
            "iat_mean": 0.05,
            "iat_std": 0.1,
            "sni": "github.com",
            "dst_port": 443,
        },
    ]

    print("\n" + "-" * 70)
    print("Classification Results:")
    print("-" * 70)

    for flow in test_flows:
        result = classifier.classify(flow)
        desc = get_priority_description(result.priority)
        print(f"\n  Flow: {flow['flow_id']}")
        print(f"    SNI: {flow.get('sni', 'N/A')}")
        print(f"    Result: {result.priority} - {desc}")
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Method: {result.method}")
        if result.probabilities:
            probs = ", ".join(
                f"{k}:{v:.2f}" for k, v in sorted(result.probabilities.items())
            )
            print(f"    Probabilities: {probs}")

    print("\n" + "-" * 70)
    print("Statistics:")
    print("-" * 70)
    stats = classifier.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Integration test complete!")
    print("=" * 70)
