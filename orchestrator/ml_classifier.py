import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from orchestrator.types import ClassificationResult

_joblib = None

logger = logging.getLogger("ml_classifier")


class MLTrafficClassifier:
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

    PORT_PRIORITY_MAP = {
        443: "P3", 5003: "P3", 8443: "P3",
        5060: "P2", 5061: "P2", 5002: "P2", 3478: "P2", 3479: "P2",
        20: "P0", 21: "P0", 22: "P0", 5000: "P0",
        80: "P1", 8080: "P1", 5001: "P1",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
        enable_fallback: bool = True,
    ):
        self.model = None
        self.label_encoder = None
        self.feature_names = self.DEFAULT_FEATURE_NAMES.copy()
        self.confidence_threshold = confidence_threshold
        self.enable_fallback = enable_fallback
        self.model_metadata = {}

        self.stats = {
            "total_classifications": 0,
            "ml_classifications": 0,
            "sni_classifications": 0,
            "port_classifications": 0,
            "default_classifications": 0,
            "low_confidence_fallbacks": 0,
        }

        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None) -> bool:
        global _joblib

        if _joblib is None:
            try:
                import joblib
                _joblib = joblib
            except ImportError:
                logger.warning("joblib not available. ML classification disabled.")
                return False

        search_paths = []
        if model_path:
            search_paths.append(Path(model_path))

        project_root = Path(__file__).parent.parent
        search_paths.extend([
            project_root / "notebooks" / "ml" / "models" / "traffic_classifier.pkl",
            project_root / "ml" / "models" / "traffic_classifier.pkl",
        ])

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

            encoder_path = model_file.parent / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = _joblib.load(encoder_path)

            feature_path = model_file.parent / "feature_names.json"
            if feature_path.exists():
                with open(feature_path) as f:
                    self.feature_names = json.load(f)

            metadata_path = model_file.parent / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.model_metadata = json.load(f)

            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _extract_features(self, flow: Dict[str, Any]) -> np.ndarray:
        features = []
        for name in self.feature_names:
            value = flow.get(name, 0)
            if value is None:
                value = 0
            features.append(float(value))
        return np.array([features])

    def _classify_ml(self, flow: Dict[str, Any]) -> Optional[ClassificationResult]:
        if self.model is None:
            return None

        try:
            X = self._extract_features(flow)
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]

            if self.label_encoder is not None:
                priority = self.label_encoder.inverse_transform([pred])[0]
                class_probas = {
                    self.label_encoder.classes_[i]: float(p)
                    for i, p in enumerate(proba)
                }
            else:
                priority = str(pred)
                class_probas = {f"class_{i}": float(p) for i, p in enumerate(proba)}

            confidence = float(max(proba))
            return ClassificationResult(
                priority=priority, confidence=confidence,
                method="ml", probabilities=class_probas,
            )
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return None

    def _classify_sni(self, flow: Dict[str, Any]) -> Optional[ClassificationResult]:
        sni = flow.get("sni", "") or flow.get("sni_domain", "")
        if not sni or sni.lower() in ("", "unknown", "none"):
            return None

        try:
            from orchestrator.sni_classifier import SNIClassifier
            classifier = SNIClassifier()
            priority = classifier.classify(sni)
            return ClassificationResult(
                priority=priority,
                confidence=0.9 if priority != "P1" else 0.5,
                method="sni",
            )
        except ImportError:
            pass

        sni_lower = sni.lower()
        if any(kw in sni_lower for kw in ["bank", "payment", "pay", "razorpay", "stripe", "paytm"]):
            return ClassificationResult(priority="P3", confidence=0.85, method="sni")
        if any(kw in sni_lower for kw in ["meet", "zoom", "teams", "webrtc", "voip", "sip"]):
            return ClassificationResult(priority="P2", confidence=0.85, method="sni")
        if any(kw in sni_lower for kw in ["cdn", "download", "update", "release", "iso", "torrent"]):
            return ClassificationResult(priority="P0", confidence=0.75, method="sni")

        return None

    def _classify_port(self, flow: Dict[str, Any]) -> ClassificationResult:
        dst_port = flow.get("dst_port", 0)
        src_port = flow.get("src_port", 0)

        if dst_port in self.PORT_PRIORITY_MAP:
            return ClassificationResult(
                priority=self.PORT_PRIORITY_MAP[dst_port], confidence=0.7, method="port",
            )
        if src_port in self.PORT_PRIORITY_MAP:
            return ClassificationResult(
                priority=self.PORT_PRIORITY_MAP[src_port], confidence=0.6, method="port",
            )
        return ClassificationResult(priority="P1", confidence=0.3, method="default")

    def classify(self, flow: Dict[str, Any]) -> ClassificationResult:
        self.stats["total_classifications"] += 1

        ml_result = self._classify_ml(flow)
        if ml_result is not None:
            if ml_result.confidence >= self.confidence_threshold:
                self.stats["ml_classifications"] += 1
                return ml_result
            else:
                self.stats["low_confidence_fallbacks"] += 1

        if self.enable_fallback:
            sni_result = self._classify_sni(flow)
            if sni_result is not None:
                self.stats["sni_classifications"] += 1
                return sni_result

        port_result = self._classify_port(flow)
        if port_result.method == "port":
            self.stats["port_classifications"] += 1
        else:
            self.stats["default_classifications"] += 1
        return port_result

    def classify_batch(self, flows: List[Dict[str, Any]]) -> List[ClassificationResult]:
        return [self.classify(flow) for flow in flows]

    def get_stats(self) -> Dict[str, Any]:
        total = max(self.stats["total_classifications"], 1)
        return {
            **self.stats,
            "ml_rate": f"{self.stats['ml_classifications'] / total:.1%}",
            "sni_rate": f"{self.stats['sni_classifications'] / total:.1%}",
            "port_rate": f"{self.stats['port_classifications'] / total:.1%}",
            "model_loaded": self.model is not None,
            "model_accuracy": self.model_metadata.get("test_metrics", {}).get("accuracy", "N/A"),
        }

    def is_model_loaded(self) -> bool:
        return self.model is not None


_classifier_instance: Optional[MLTrafficClassifier] = None


def get_classifier() -> MLTrafficClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = MLTrafficClassifier()
    return _classifier_instance
