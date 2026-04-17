import os
import sys
import time
import logging
import re
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.stubs import update_link, predict
from orchestrator.policy_engine import PolicyEngine
from collector.scrape_prometheus import PrometheusScraper
from collector.scrape_ovs import OVSScraper
from collector.packet_capture import PacketCapture
from collector.build_datasets import DatasetBuilder, label_flow_by_port
from prometheus_client import start_http_server, Gauge as PromGauge, CollectorRegistry, REGISTRY

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orchestrator")

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 10))
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
DATA_DIR = os.environ.get("DATA_DIR", "/app/data/processed")
CAPTURE_INTERFACES = os.environ.get("CAPTURE_INTERFACES", "s1-eth1,s1-eth2").split(",")
METRICS_PORT = 8000

# Prometheus metrics — use a dedicated registry to avoid conflicts
_metrics_registry = CollectorRegistry()

FLOW_BYTES = PromGauge(
    "sdn_flow_bytes", "Bytes per flow",
    ["flow_id", "src_port", "dst_port", "protocol"],
    registry=_metrics_registry,
)
FLOW_PACKETS = PromGauge(
    "sdn_flow_packets", "Packets per flow",
    ["flow_id", "src_port", "dst_port", "protocol"],
    registry=_metrics_registry,
)
PORT_BYTES_RX = PromGauge(
    "sdn_port_rx_bytes", "Received bytes per port",
    ["dp_name", "port"],
    registry=_metrics_registry,
)
PORT_BYTES_TX = PromGauge(
    "sdn_port_tx_bytes", "Transmitted bytes per port",
    ["dp_name", "port"],
    registry=_metrics_registry,
)


class Orchestrator:
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.prometheus = PrometheusScraper(PROMETHEUS_URL)
        self.ovs = OVSScraper("s1")
        self.dataset_builder = DatasetBuilder(DATA_DIR)
        self.running = True

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

        logger.info(f"Starting Prometheus exporter on port {METRICS_PORT}")
        start_http_server(METRICS_PORT, registry=_metrics_registry)

    def start_captures(self):
        for pc in self.captures:
            try:
                pc.start()
            except Exception as e:
                logger.error(f"Failed to start capture on {pc.interface}: {e}")

    def stop_captures(self):
        for pc in self.captures:
            pc.stop()

    def collect_stats(self) -> Dict[str, Any]:
        try:
            port_stats = self.prometheus.get_faucet_port_stats()

            for switch_name in ["s1", "s2", "s3", "s4", "s5", "s6"]:
                scraper = OVSScraper(switch_name)
                ovs_flows = scraper.get_flow_stats()

                for f in ovs_flows:
                    if "tcp" in f.match or "udp" in f.match:
                        sp_match = re.search(r"tp_src=(\d+)", f.match)
                        dp_match = re.search(r"tp_dst=(\d+)", f.match)
                        src_port = int(sp_match.group(1)) if sp_match else 0
                        dst_port = int(dp_match.group(1)) if dp_match else 0
                        proto = "tcp" if "tcp" in f.match else "udp"
                        flow_id = f"{switch_name}_{f.cookie}_{src_port}_{dst_port}"

                        FLOW_BYTES.labels(flow_id, src_port, dst_port, proto).set(f.n_bytes)
                        FLOW_PACKETS.labels(flow_id, src_port, dst_port, proto).set(f.n_packets)

                ovs_ports = scraper.get_port_stats()
                for p in ovs_ports:
                    PORT_BYTES_RX.labels(switch_name, p.port_no).set(p.rx_bytes)
                    PORT_BYTES_TX.labels(switch_name, p.port_no).set(p.tx_bytes)

            # Prefer Scapy flows if available, fall back to OVS
            flows = []
            scapy_flows = []
            for pc in self.captures:
                scapy_flows.extend(pc.get_flow_stats())

            if scapy_flows:
                for f in scapy_flows:
                    flow_id = f"scapy_{f['src_ip']}_{f['dst_ip']}_{f['src_port']}_{f['dst_port']}"
                    f["flow_id"] = flow_id
                    label = label_flow_by_port(f["dst_port"], f["src_port"])
                    record = self.dataset_builder.build_flow_record(f, label=label)
                    self.dataset_builder.add_flow(record)

                    f["bytes"] = f["total_bytes"] if "total_bytes" in f else f.get("byte_count", 0)
                    f["packets"] = f["total_packets"] if "total_packets" in f else f.get("packet_count", 0)
                    flows.append(f)
            else:
                self.ovs = OVSScraper("s1")
                ovs_flows = self.ovs.get_flow_stats()

                for f in ovs_flows:
                    if "tcp" in f.match or "udp" in f.match:
                        sp_match = re.search(r"tp_src=(\d+)", f.match)
                        dp_match = re.search(r"tp_dst=(\d+)", f.match)
                        src_port = int(sp_match.group(1)) if sp_match else 0
                        dst_port = int(dp_match.group(1)) if dp_match else 0
                        flow_id = f"{f.cookie}_{src_port}_{dst_port}"

                        flows.append({
                            "flow_id": flow_id,
                            "src_ip": "unknown",
                            "dst_ip": "unknown",
                            "src_port": src_port,
                            "dst_port": dst_port,
                            "protocol": 6 if "tcp" in f.match else 17,
                            "bytes": f.n_bytes,
                            "packets": f.n_packets,
                            "duration": f.duration,
                        })

            links = []
            for p in port_stats:
                links.append({
                    "switch": p["dp_name"],
                    "port": int(p["port"]),
                    "bytes_rx": p.get("rx_bytes", 0),
                    "bytes_tx": p.get("tx_bytes", 0),
                })

            return {"flows": flows, "links": links}
        except Exception as e:
            logger.error(f"Failed to collect stats: {e}")
            return {"flows": [], "links": []}

    def update_predictions(self, links: list) -> Dict[str, Dict]:
        predictions = {}
        for link in links:
            switch = link["switch"]
            port = link["port"]
            bytes_total = link.get("bytes_rx", 0) + link.get("bytes_tx", 0)
            update_link(switch, port, bytes_total)
            prediction = predict(switch, port)
            predictions[f"{switch}:{port}"] = prediction

            if prediction["predicted_congestion"]:
                logger.warning(
                    f"Congestion predicted on {switch}:{port} "
                    f"(util={prediction['current_utilization']:.1%})"
                )
        return predictions

    def run_once(self) -> None:
        logger.debug("Running orchestrator iteration")
        stats = self.collect_stats()

        # Classification is handled by PolicyEngine.apply() — no separate classify step
        predictions = self.update_predictions(stats["links"])
        decisions = self.policy_engine.apply(stats["flows"], predictions)

        for decision in decisions:
            logger.info(f"POLICY: {decision}")

    def run(self) -> None:
        logger.info("Starting orchestrator")
        logger.info(f"Poll interval: {POLL_INTERVAL}s")
        logger.info(f"ML enabled: {self.policy_engine.ml_classifier is not None}")

        self.start_captures()

        while self.running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Error in orchestrator loop: {e}")
            time.sleep(POLL_INTERVAL)

    def stop(self) -> None:
        self.running = False
        self.stop_captures()
        logger.info("Orchestrator stopped")


def main():
    orchestrator = Orchestrator()
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        orchestrator.stop()


if __name__ == "__main__":
    main()
