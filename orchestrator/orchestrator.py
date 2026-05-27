import os
import sys
import time
import logging
import re
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.stubs import update_link, predict
from orchestrator.policy_engine import PolicyEngine
from orchestrator.types import (
    POLL_INTERVAL,
    PROMETHEUS_URL,
    DATA_DIR,
    CAPTURE_INTERFACES,
    ORCHESTRATOR_METRICS_PORT,
    classify_by_port,
)
from collector.scrape_prometheus import PrometheusScraper
from collector.scrape_ovs import OVSScraper
from collector.build_datasets import DatasetBuilder
from prometheus_client import start_http_server, Gauge as PromGauge, CollectorRegistry

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orchestrator")

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


# Lazy imports — Scapy needs raw sockets (fails in containers/sandboxes)
_PacketCapture = None  # set by _get_packet_capture_cls()


def _get_packet_capture_cls():
    """Lazily import PacketCapture so Scapy isn't loaded at module level."""
    global _PacketCapture
    if _PacketCapture is not None:
        return _PacketCapture
    try:
        from collector.packet_capture import PacketCapture as PC
        _PacketCapture = PC
    except (ImportError, PermissionError) as exc:
        logger.warning(f"Packet capture not available (Scapy init failed): {exc}")
        _PacketCapture = None
    return _PacketCapture


def _build_captures(interface_spec: str) -> list:
    """Build packet-capture instances from a comma-separated interface list.

    Returns an empty list if Scapy is unavailable.
    """
    cls = _get_packet_capture_cls()
    if cls is None:
        return []

    captures = []
    for iface in interface_spec.split(","):
        iface = iface.strip()
        if iface:
            try:
                captures.append(cls(interface=iface))
                logger.info(f"Initialized packet capture on {iface}")
            except Exception as e:
                logger.warning(f"Failed to init capture on {iface}: {e}")
    return captures


class Orchestrator:
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.prometheus = PrometheusScraper(PROMETHEUS_URL)
        self.dataset_builder = DatasetBuilder(DATA_DIR)
        self.running = True

        # Cache OVS scraper instances — avoid spawning ovs-ofctl 12× per poll
        self._ovs_scrapers: Dict[str, OVSScraper] = {}

        self.captures: List[Any] = _build_captures(CAPTURE_INTERFACES)

        logger.info(f"Starting Prometheus exporter on port {ORCHESTRATOR_METRICS_PORT}")
        start_http_server(ORCHESTRATOR_METRICS_PORT, registry=_metrics_registry)

    # --- helpers -----------------------------------------------------------

    def _get_scraper(self, switch_name: str) -> OVSScraper:
        """Return a cached :class:`OVSScraper` for *switch_name*."""
        if switch_name not in self._ovs_scrapers:
            self._ovs_scrapers[switch_name] = OVSScraper(switch_name)
        return self._ovs_scrapers[switch_name]

    @staticmethod
    def _parse_ports_from_match(match_str: str):
        """Extract ``(src_port, dst_port, protocol)`` from an OVS match string."""
        sp_match = re.search(r"tp_src=(\d+)", match_str)
        dp_match = re.search(r"tp_dst=(\d+)", match_str)
        src_port = int(sp_match.group(1)) if sp_match else 0
        dst_port = int(dp_match.group(1)) if dp_match else 0
        proto = "udp" if "udp" in match_str else "tcp"
        return src_port, dst_port, proto

    # --- public API --------------------------------------------------------

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
                scraper = self._get_scraper(switch_name)
                ovs_flows = scraper.get_flow_stats()

                for f in ovs_flows:
                    if "tcp" in f.match or "udp" in f.match:
                        src_port, dst_port, proto = self._parse_ports_from_match(f.match)
                        flow_id = f"{switch_name}_{f.cookie}_{src_port}_{dst_port}"
                        FLOW_BYTES.labels(flow_id, src_port, dst_port, proto).set(f.n_bytes)
                        FLOW_PACKETS.labels(flow_id, src_port, dst_port, proto).set(f.n_packets)

                ovs_ports = scraper.get_port_stats()
                for p in ovs_ports:
                    PORT_BYTES_RX.labels(switch_name, p.port_no).set(p.rx_bytes)
                    PORT_BYTES_TX.labels(switch_name, p.port_no).set(p.tx_bytes)

            # Prefer Scapy flows if available, fall back to OVS
            flows: List[Dict] = []
            scapy_flows: List[Dict] = []
            for pc in self.captures:
                scapy_flows.extend(pc.get_flow_stats())

            if scapy_flows:
                for f in scapy_flows:
                    flow_id = f"scapy_{f['src_ip']}_{f['dst_ip']}_{f['src_port']}_{f['dst_port']}"
                    f["flow_id"] = flow_id
                    label = classify_by_port(f["dst_port"], f["src_port"])
                    record = self.dataset_builder.build_flow_record(f, label=label)
                    self.dataset_builder.add_flow(record)

                    f["bytes"] = f.get("total_bytes", f.get("byte_count", 0))
                    f["packets"] = f.get("total_packets", f.get("packet_count", 0))
                    flows.append(f)
            else:
                scraper = self._get_scraper("s1")
                ovs_flows = scraper.get_flow_stats()

                for f in ovs_flows:
                    if "tcp" in f.match or "udp" in f.match:
                        src_port, dst_port, proto = self._parse_ports_from_match(f.match)
                        flow_id = f"{f.cookie}_{src_port}_{dst_port}"

                        flows.append({
                            "flow_id": flow_id,
                            "src_ip": "unknown",
                            "dst_ip": "unknown",
                            "src_port": src_port,
                            "dst_port": dst_port,
                            "protocol": 6 if proto == "tcp" else 17,
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
