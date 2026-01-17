"""
Orchestrator for SDN ML Traffic Management
Main control loop that coordinates classification, prediction, and policy enforcement.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.stubs import classify_flow, predict, update_link, get_all_predictions
from orchestrator.policy_engine import PolicyEngine

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orchestrator")

# Configuration
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 10))  # seconds
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/ml/models")


class Orchestrator:
    """
    Main orchestrator that coordinates:
    1. Telemetry collection
    2. Flow classification
    3. Congestion prediction
    4. Policy enforcement
    """

    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.use_ml_models = False  # Use stubs until models are trained
        self.running = True

        # Try to load ML models
        self._try_load_models()

    def _try_load_models(self) -> None:
        """Attempt to load trained ML models"""
        classifier_path = os.path.join(MODEL_PATH, "classifier.pkl")
        predictor_path = os.path.join(MODEL_PATH, "predictor.pkl")

        if os.path.exists(classifier_path) and os.path.exists(predictor_path):
            try:
                import joblib

                self.classifier = joblib.load(classifier_path)
                self.predictor = joblib.load(predictor_path)
                self.use_ml_models = True
                logger.info("Loaded trained ML models")
            except Exception as e:
                logger.warning(f"Failed to load ML models: {e}")
                logger.info("Using hardcoded stubs instead")
        else:
            logger.info("ML models not found, using hardcoded stubs")

    def collect_stats(self) -> Dict[str, Any]:
        """
        Collect current network statistics.

        MVP: Uses simulated data
        Phase 3+: Will use Prometheus/OVS scraping
        """
        # TODO: Replace with actual Prometheus/OVS scraping in Phase 3

        # Simulated stats for MVP
        return {
            "flows": [
                {
                    "flow_id": 1,
                    "src_ip": "10.0.0.1",
                    "dst_ip": "10.0.0.2",
                    "src_port": 12345,
                    "dst_port": 5001,
                    "protocol": "tcp",
                    "bytes": 1000000,
                    "packets": 1000,
                    "duration": 10,
                }
            ],
            "links": [
                {
                    "switch": "s1",
                    "port": 1,
                    "bytes_rx": 5000000,
                    "bytes_tx": 5000000,
                }
            ],
        }

    def classify_flows(self, flows: list) -> list:
        """Classify all flows by priority"""
        classified = []

        for flow in flows:
            # Extract features
            features = {
                "src_port": flow.get("src_port", 0),
                "dst_port": flow.get("dst_port", 0),
                "protocol": flow.get("protocol", "tcp"),
                "bytes_per_sec": flow.get("bytes", 0) / max(1, flow.get("duration", 1)),
                "packet_size_avg": flow.get("bytes", 0)
                / max(1, flow.get("packets", 1)),
            }

            # Classify
            if self.use_ml_models:
                # TODO: Use ML model
                priority = classify_flow(features)  # Fallback to stub for now
            else:
                priority = classify_flow(features)

            flow["priority"] = priority
            classified.append(flow)

            logger.debug(f"Flow {flow['flow_id']}: {priority}")

        return classified

    def update_predictions(self, links: list) -> Dict[str, Dict]:
        """Update link statistics and get predictions"""
        predictions = {}

        for link in links:
            switch = link["switch"]
            port = link["port"]
            bytes_total = link.get("bytes_rx", 0) + link.get("bytes_tx", 0)

            # Update link state
            update_link(switch, port, bytes_total)

            # Get prediction
            prediction = predict(switch, port)
            predictions[f"{switch}:{port}"] = prediction

            if prediction["predicted_congestion"]:
                logger.warning(
                    f"Congestion predicted on {switch}:{port} "
                    f"(util={prediction['current_utilization']:.1%})"
                )

        return predictions

    def run_once(self) -> None:
        """Single iteration of the orchestrator loop"""
        logger.debug("Running orchestrator iteration")

        # 1. Collect stats
        stats = self.collect_stats()

        # 2. Classify flows
        classified_flows = self.classify_flows(stats["flows"])

        # 3. Predict congestion
        predictions = self.update_predictions(stats["links"])

        # 4. Apply policies
        decisions = self.policy_engine.apply(classified_flows, predictions)

        # 5. Log decisions
        for decision in decisions:
            logger.info(f"POLICY: {decision}")

    def run(self) -> None:
        """Main orchestrator loop"""
        logger.info("Starting orchestrator")
        logger.info(f"Poll interval: {POLL_INTERVAL}s")
        logger.info(f"Using ML models: {self.use_ml_models}")

        while self.running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Error in orchestrator loop: {e}")

            time.sleep(POLL_INTERVAL)

    def stop(self) -> None:
        """Stop the orchestrator"""
        self.running = False
        logger.info("Orchestrator stopped")


def main():
    """Entry point"""
    orchestrator = Orchestrator()

    try:
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        orchestrator.stop()


if __name__ == "__main__":
    main()
