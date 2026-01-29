"""
Orchestrator for SDN ML Traffic Management
Main control loop that coordinates classification, prediction, and policy enforcement.
"""

import os
import sys
import time
import logging
import re
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.stubs import classify_flow, predict, update_link, get_all_predictions
from orchestrator.policy_engine import PolicyEngine
from collector.scrape_prometheus import PrometheusScraper
from collector.scrape_ovs import OVSScraper
from collector.packet_capture import PacketCapture
from collector.build_datasets import DatasetBuilder, label_flow_by_port
from prometheus_client import start_http_server, Gauge as PromGauge, REGISTRY

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
DATA_DIR = os.environ.get("DATA_DIR", "/app/data/processed")
CAPTURE_INTERFACES = os.environ.get("CAPTURE_INTERFACES", "s1-eth1,s1-eth2").split(",")
METRICS_PORT = 8000


def get_or_create_gauge(name, doc, labels):
    """Get existing gauge or create new one to avoid duplicates"""
    # Check if metric already exists in registry
    for collector in REGISTRY._collector_to_names.keys():
        if hasattr(collector, "_name") and collector._name == name:
            return collector
    # Fallback: create new
    try:
        return PromGauge(name, doc, labels)
    except ValueError:
        # If race condition or other issue, try to find it again or ignore
        return PromGauge(name, doc, labels)


# Prometheus Metrics
FLOW_BYTES = get_or_create_gauge(
    "sdn_flow_bytes", "Bytes per flow", ["flow_id", "src_port", "dst_port", "protocol"]
)
FLOW_PACKETS = get_or_create_gauge(
    "sdn_flow_packets",
    "Packets per flow",
    ["flow_id", "src_port", "dst_port", "protocol"],
)
PORT_BYTES_RX = get_or_create_gauge(
    "sdn_port_rx_bytes", "Received bytes per port", ["dp_name", "port"]
)
PORT_BYTES_TX = get_or_create_gauge(
    "sdn_port_tx_bytes", "Transmitted bytes per port", ["dp_name", "port"]
)


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
        self.prometheus = PrometheusScraper(PROMETHEUS_URL)
        self.ovs = OVSScraper("s1")  # MVP bridge name
        self.dataset_builder = DatasetBuilder(DATA_DIR)
        self.use_ml_models = False  # Use stubs until models are trained
        self.running = True

        # Initialize Packet Captures
        self.captures = []
        try:
            for iface in CAPTURE_INTERFACES:
                iface = iface.strip()
                if iface:
                    pc = PacketCapture(interface=iface)
                    self.captures.append(pc)
                    logger.info(f"Initialized packet capture on {iface}")
        except Exception as e:
            logger.warning(f"Failed to initialize packet captures: {e}")

        # Start Prometheus Exporter
        logger.info(f"Starting Prometheus exporter on port {METRICS_PORT}")
        start_http_server(METRICS_PORT)

        # Try to load ML models
        self._try_load_models()

    def start_captures(self):
        """Start all packet captures"""
        for pc in self.captures:
            try:
                pc.start()
            except Exception as e:
                logger.error(f"Failed to start capture on {pc.interface}: {e}")

    def stop_captures(self):
        """Stop all packet captures"""
        for pc in self.captures:
            pc.stop()

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

        Phase 3: Uses Prometheus and OVS scraping
        """
        try:
            # 1. Collect port stats and utilization from Prometheus
            port_stats = self.prometheus.get_faucet_port_stats()

            # 2. Collect flow stats from OVS
            # Loop through all switches for metrics
            all_flows = []
            for switch_name in ["s1", "s2", "s3", "s4", "s5", "s6"]:
                scraper = OVSScraper(switch_name)
                ovs_flows = scraper.get_flow_stats()

                for f in ovs_flows:
                    # Update Flow Metrics
                    if "tcp" in f.match or "udp" in f.match:
                        sp_match = re.search(r"tp_src=(\d+)", f.match)
                        dp_match = re.search(r"tp_dst=(\d+)", f.match)

                        src_port = int(sp_match.group(1)) if sp_match else 0
                        dst_port = int(dp_match.group(1)) if dp_match else 0
                        proto = "tcp" if "tcp" in f.match else "udp"

                        flow_id = f"{switch_name}_{f.cookie}_{src_port}_{dst_port}"

                        FLOW_BYTES.labels(flow_id, src_port, dst_port, proto).set(
                            f.n_bytes
                        )
                        FLOW_PACKETS.labels(flow_id, src_port, dst_port, proto).set(
                            f.n_packets
                        )

                # Update Port Metrics (if OVSScraper supports port stats)
                ovs_ports = scraper.get_port_stats()
                for p in ovs_ports:
                    PORT_BYTES_RX.labels(switch_name, p.port_no).set(p.rx_bytes)
                    PORT_BYTES_TX.labels(switch_name, p.port_no).set(p.tx_bytes)

            # Re-use s1 flows for policy engine (for now)
            # In real system, we'd aggregate
            self.ovs = OVSScraper("s1")
            ovs_flows = self.ovs.get_flow_stats()

            # Map OVS flows to internal flow format
            flows = []

            # Phase 3.5: Prefer Scapy flows if available
            scapy_flows = []
            for pc in self.captures:
                scapy_flows.extend(pc.get_flow_stats())

            if scapy_flows:
                # Use Scapy flows which have rich features
                for f in scapy_flows:
                    # Construct Flow ID
                    flow_id = f"scapy_{f['src_ip']}_{f['dst_ip']}_{f['src_port']}_{f['dst_port']}"
                    f["flow_id"] = flow_id

                    # Add to dataset
                    # For labeling, we use port-based heuristics for now (Phase 2/3 style)
                    # Real labeling will come from controlled experiments or SNI
                    label = label_flow_by_port(f["dst_port"], f["src_port"])

                    record = self.dataset_builder.build_flow_record(f, label=label)

                    self.dataset_builder.add_flow(record)

                    # Normalize for policy engine
                    f["bytes"] = f["total_bytes"]
                    f["packets"] = f["total_packets"]
                    # Protocol mapping: 6->tcp, 17->udp
                    f["protocol_str"] = "tcp" if f["protocol"] == 6 else "udp"
                    if f["protocol"] == 17:
                        f["protocol_str"] = "udp"

                    flows.append(f)

                logger.debug(f"Collected {len(flows)} flows from Scapy")

            else:
                # Fallback to OVS flows (Phase 3)
                for f in ovs_flows:
                    # Only process flows with specific match criteria (e.g. TCP/UDP)
                    if "tcp" in f.match or "udp" in f.match:
                        # Extract ports from match string (heuristic)
                        # Example match: "tcp,in_port=1,tp_src=12345,tp_dst=5001"
                        src_port = 0
                        dst_port = 0

                        sp_match = re.search(r"tp_src=(\d+)", f.match)
                        dp_match = re.search(r"tp_dst=(\d+)", f.match)

                        if sp_match:
                            src_port = int(sp_match.group(1))
                        if dp_match:
                            dst_port = int(dp_match.group(1))

                        # Create a more unique flow ID for tracking
                        flow_id = f"{f.cookie}_{src_port}_{dst_port}"

                        flows.append(
                            {
                                "flow_id": flow_id,
                                "src_ip": "unknown",
                                "dst_ip": "unknown",
                                "src_port": src_port,
                                "dst_port": dst_port,
                                "protocol": "tcp" if "tcp" in f.match else "udp",
                                "bytes": f.n_bytes,
                                "packets": f.n_packets,
                                "duration": f.duration,
                            }
                        )

            # Map ports to internal links format
            links = []
            for p in port_stats:
                links.append(
                    {
                        "switch": p["dp_name"],
                        "port": int(p["port"]),
                        "bytes_rx": p.get("rx_bytes", 0),
                        "bytes_tx": p.get("tx_bytes", 0),
                    }
                )

            return {"flows": flows, "links": links}
        except Exception as e:
            logger.error(f"Failed to collect stats: {e}")
            # Fallback to empty stats
            return {"flows": [], "links": []}

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

        # Start packet captures
        self.start_captures()

        while self.running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Error in orchestrator loop: {e}")

            time.sleep(POLL_INTERVAL)

    def stop(self) -> None:
        """Stop the orchestrator"""
        self.running = False
        self.stop_captures()
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
