"""
SDN Orchestrator — Main control loop

Collects telemetry, runs ML classification + congestion prediction,
and enforces QoS/routing policies via the PolicyEngine.

Pipeline: Collectors (async buf) → Feature Extraction → ML Inference
          → Policy Engine → QoS Enforcer

Health metrics are exported on port 8000 for Prometheus scraping.
"""

import os
import sys
import time
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.stubs import update_link, predict
from orchestrator.policy_engine import PolicyEngine
from collector.scrape_prometheus import PrometheusScraper
from collector.scrape_ovs import OVSScraper
# PacketCapture imported lazily in _init_captures()
from collector.build_datasets import DatasetBuilder, label_flow_by_port
from prometheus_client import start_http_server, Gauge as PromGauge, Counter, Histogram, CollectorRegistry

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orchestrator")

# =============================================================================
# Configuration
# =============================================================================
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 10))
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
DATA_DIR = os.environ.get("DATA_DIR", "/app/data/processed")
CAPTURE_INTERFACES = os.environ.get("CAPTURE_INTERFACES", "s1-eth1,s1-eth2").split(",")
METRICS_PORT = 8000
SWITCH_NAMES = ["s1", "s2", "s3", "s4", "s5", "s6"]

# Circuit-breaker settings
MAX_COLLECTOR_FAILURES = 3
COLLECTOR_RETRY_INTERVAL = 30  # seconds before retrying a failed collector

# =============================================================================
# Prometheus metrics (dedicated registry)
# =============================================================================
_metrics_registry = CollectorRegistry()

# Health / liveness
SDN_UP = PromGauge("sdn_up", "Orchestrator is running", registry=_metrics_registry)
SDN_COLLECTOR_UP = PromGauge(
    "sdn_collector_up", "Collector health", ["name"], registry=_metrics_registry
)
SDN_ML_MODEL_LOADED = PromGauge(
    "sdn_ml_model_loaded", "ML model loaded (1=yes, 0=no)", ["model"], registry=_metrics_registry
)
SDN_PIPELINE_ITERATIONS = Counter(
    "sdn_pipeline_iterations_total", "Pipeline iterations", ["status"], registry=_metrics_registry,
)
SDN_PIPELINE_LATENCY = Histogram(
    "sdn_pipeline_latency_seconds", "Pipeline stage latency",
    ["stage"], buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5], registry=_metrics_registry,
)

# Flow metrics
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

SDN_CONCURRENT_FLOWS = PromGauge(
    "sdn_concurrent_flows", "Current number of tracked flows",
    registry=_metrics_registry,
)
SDN_POLICY_DECISIONS = Counter(
    "sdn_policy_decisions_total", "Policy decisions made",
    ["priority", "action"], registry=_metrics_registry,
)


class Orchestrator:
    """Main orchestrator: collect → predict → decide → enforce."""

    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.prometheus = PrometheusScraper(PROMETHEUS_URL)
        self.dataset_builder = DatasetBuilder(DATA_DIR)
        self.running = True

        # Cache OVS scrapers per switch (don't recreate 6 per poll)
        self._ovs_scrapers = {name: OVSScraper(name) for name in SWITCH_NAMES}
        self._ovs_scraper_s1 = self._ovs_scrapers["s1"]

        # Circuit-breaker state for collectors
        self._collector_failures: Dict[str, int] = {}
        self._collector_cooldown: Dict[str, float] = {}

        # Packet captures (optional, requires root / Mininet namespace)
        self.captures: List[PacketCapture] = []
        self._init_captures()

        # Seed health metrics
        SDN_UP.set(1)
        SDN_COLLECTOR_UP.labels(name="prometheus").set(0)
        SDN_COLLECTOR_UP.labels(name="ovs").set(0)
        SDN_COLLECTOR_UP.labels(name="scapy").set(0)
        SDN_ML_MODEL_LOADED.labels(model="classifier").set(
            1 if self.policy_engine.ml_classifier is not None else 0
        )

        logger.info(f"Starting Prometheus exporter on port {METRICS_PORT}")
        start_http_server(METRICS_PORT, registry=_metrics_registry)

    def _init_captures(self):
        """Initialize packet captures (non-fatal if unavailable)."""
        from collector.packet_capture import PacketCapture  # lazy import (needs root)
        for iface in CAPTURE_INTERFACES:
            iface = iface.strip()
            if iface:
                try:
                    pc = PacketCapture(interface=iface)
                    self.captures.append(pc)
                    logger.info(f"Initialized packet capture on {iface}")
                except Exception as e:
                    logger.warning(f"Failed to init packet capture on {iface}: {e}")

    def _is_collector_available(self, name: str) -> bool:
        """Circuit breaker: check if a collector should be retried."""
        now = time.time()
        if name in self._collector_cooldown:
            if now < self._collector_cooldown[name]:
                return False
            # Cooldown expired, clear failure count
            self._collector_failures.pop(name, None)
            self._collector_cooldown.pop(name, None)
        return True

    def _record_collector_failure(self, name: str):
        """Increment circuit-breaker counter for a collector."""
        self._collector_failures[name] = self._collector_failures.get(name, 0) + 1
        if self._collector_failures[name] >= MAX_COLLECTOR_FAILURES:
            cool_until = time.time() + COLLECTOR_RETRY_INTERVAL
            self._collector_cooldown[name] = cool_until
            logger.warning(
                f"Collector '{name}' circuit opened — "
                f"retrying in {COLLECTOR_RETRY_INTERVAL}s"
            )

    def start_captures(self):
        for pc in self.captures:
            try:
                pc.start()
                SDN_COLLECTOR_UP.labels(name="scapy").set(1)
            except Exception as e:
                logger.error(f"Failed to start capture on {pc.interface}: {e}")

    def stop_captures(self):
        for pc in self.captures:
            pc.stop()

    # -------------------------------------------------------------------------
    # Pipeline stage 1: Collect
    # -------------------------------------------------------------------------
    def _collect_flows(self) -> List[Dict[str, Any]]:
        """Collect flow stats from Scapy (preferred) or OVS fallback."""
        t0 = time.time()

        # Try Scapy captures first
        scapy_flows: List[Dict] = []
        for pc in self.captures:
            scapy_flows.extend(pc.get_flow_stats())

        if scapy_flows:
            flows = []
            for f in scapy_flows:
                flow_id = (
                    f"scapy_{f['src_ip']}_{f['dst_ip']}"
                    f"_{f['src_port']}_{f['dst_port']}"
                )
                f["flow_id"] = flow_id
                label = label_flow_by_port(f["dst_port"], f["src_port"])
                record = self.dataset_builder.build_flow_record(f, label=label)
                self.dataset_builder.add_flow(record)

                flows.append({
                    "flow_id": flow_id,
                    "src_ip": f.get("src_ip", ""),
                    "dst_ip": f.get("dst_ip", ""),
                    "src_port": f.get("src_port", 0),
                    "dst_port": f.get("dst_port", 0),
                    "protocol": f.get("protocol", 6),
                    "bytes": f.get("total_bytes", f.get("byte_count", 0)),
                    "packets": f.get("total_packets", f.get("packet_count", 0)),
                    "duration": f.get("duration", 1.0),
                    "sni": f.get("sni", ""),
                })

            self._record_collector_success("scapy")
            SDN_COLLECTOR_UP.labels(name="scapy").set(1)
            SDN_CONCURRENT_FLOWS.set(len(flows))
            SDN_PIPELINE_LATENCY.labels(stage="collect_flows").observe(time.time() - t0)
            return flows

        # Fallback: OVS stats from s1
        if not self._is_collector_available("ovs"):
            logger.debug("OVS collector in cooldown — skipping flow collection")
            SDN_PIPELINE_LATENCY.labels(stage="collect_flows").observe(time.time() - t0)
            return []

        flows = []
        try:
            ovs_flows = self._ovs_scraper_s1.get_flow_stats()
            for f in ovs_flows:
                if "tcp" not in f.match and "udp" not in f.match:
                    continue
                sp_match = re.search(r"tp_src=(\d+)", f.match)
                dp_match = re.search(r"tp_dst=(\d+)", f.match)
                src_port = int(sp_match.group(1)) if sp_match else 0
                dst_port = int(dp_match.group(1)) if dp_match else 0
                proto = 6 if "tcp" in f.match else 17
                flow_id = f"ovs_{f.cookie}_{src_port}_{dst_port}"

                flows.append({
                    "flow_id": flow_id,
                    "src_ip": "unknown",
                    "dst_ip": "unknown",
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": proto,
                    "bytes": f.n_bytes,
                    "packets": f.n_packets,
                    "duration": f.duration,
                })

            self._record_collector_success("ovs")
            SDN_COLLECTOR_UP.labels(name="ovs").set(1)
        except Exception as e:
            logger.error(f"OVS flow collection failed: {e}")
            self._record_collector_failure("ovs")
            SDN_COLLECTOR_UP.labels(name="ovs").set(0)

        SDN_CONCURRENT_FLOWS.set(len(flows))
        SDN_PIPELINE_LATENCY.labels(stage="collect_flows").observe(time.time() - t0)
        return flows

    def _collect_port_stats(self) -> List[Dict[str, Any]]:
        """Collect port stats from all switches."""
        t0 = time.time()

        # Try Prometheus first
        if self._is_collector_available("prometheus"):
            try:
                port_stats = self.prometheus.get_faucet_port_stats()
                links = []
                for p in port_stats:
                    links.append({
                        "switch": p["dp_name"],
                        "port": int(p["port"]),
                        "bytes_rx": p.get("rx_bytes", 0),
                        "bytes_tx": p.get("tx_bytes", 0),
                    })
                self._record_collector_success("prometheus")
                SDN_COLLECTOR_UP.labels(name="prometheus").set(1)

                # Also export OVS metrics for all switches
                for switch_name, scraper in self._ovs_scrapers.items():
                    try:
                        ovs_ports = scraper.get_port_stats()
                        for p in ovs_ports:
                            PORT_BYTES_RX.labels(switch_name, p.port_no).set(p.rx_bytes)
                            PORT_BYTES_TX.labels(switch_name, p.port_no).set(p.tx_bytes)
                    except Exception:
                        pass  # non-critical

                SDN_PIPELINE_LATENCY.labels(stage="collect_ports").observe(time.time() - t0)
                return links
            except Exception as e:
                logger.warning(f"Prometheus scrape failed: {e}")
                self._record_collector_failure("prometheus")
                SDN_COLLECTOR_UP.labels(name="prometheus").set(0)

        # Fallback: OVS port stats
        links = []
        for switch_name, scraper in self._ovs_scrapers.items():
            try:
                ovs_ports = scraper.get_port_stats()
                for p in ovs_ports:
                    PORT_BYTES_RX.labels(switch_name, p.port_no).set(p.rx_bytes)
                    PORT_BYTES_TX.labels(switch_name, p.port_no).set(p.tx_bytes)
                    links.append({
                        "switch": switch_name,
                        "port": p.port_no,
                        "bytes_rx": p.rx_bytes,
                        "bytes_tx": p.tx_bytes,
                    })
            except Exception as e:
                logger.debug(f"OVS port stats for {switch_name}: {e}")

        SDN_PIPELINE_LATENCY.labels(stage="collect_ports").observe(time.time() - t0)
        return links

    def _record_collector_success(self, name: str):
        """Reset circuit-breaker on success."""
        self._collector_failures.pop(name, None)
        self._collector_cooldown.pop(name, None)

    # -------------------------------------------------------------------------
    # Pipeline stage 2: Predict
    # -------------------------------------------------------------------------
    def _update_predictions(self, links: list) -> Dict[str, Dict]:
        """Feed link stats to the congestion predictor."""
        t0 = time.time()
        predictions: Dict[str, Dict] = {}
        for link in links:
            switch = link["switch"]
            port = link["port"]
            bytes_total = link.get("bytes_rx", 0) + link.get("bytes_tx", 0)
            try:
                update_link(switch, port, bytes_total)
                prediction = predict(switch, port)
                predictions[f"{switch}:{port}"] = prediction

                if prediction.get("predicted_congestion"):
                    logger.warning(
                        "Congestion predicted on %s:%d (util=%.1f%%)",
                        switch, port, prediction.get("current_utilization", 0) * 100,
                    )
            except Exception as e:
                logger.error("Prediction failed for %s:%d: %s", switch, port, e)

        SDN_PIPELINE_LATENCY.labels(stage="predict").observe(time.time() - t0)
        return predictions

    # -------------------------------------------------------------------------
    # Pipeline stage 3: Run one iteration
    # -------------------------------------------------------------------------
    def run_once(self) -> None:
        """Execute one full pipeline iteration."""
        logger.debug("Running orchestrator iteration")
        t_start = time.time()

        # 1) Collect
        flows = self._collect_flows()
        links = self._collect_port_stats()

        # 2) Predict
        predictions = self._update_predictions(links)

        # 3) Decide & Enforce
        decisions = self.policy_engine.apply(flows, predictions)

        for decision in decisions:
            logger.info(
                "POLICY: flow=%s priority=%s action=%s reason=%s",
                decision.flow_id, decision.priority,
                decision.action.value, decision.reason,
            )
            SDN_POLICY_DECISIONS.labels(
                priority=decision.priority,
                action=decision.action.value,
            ).inc()

        iteration_time = time.time() - t_start
        SDN_PIPELINE_LATENCY.labels(stage="total").observe(iteration_time)
        SDN_PIPELINE_ITERATIONS.labels(status="success").inc()

        logger.debug(
            "Iteration complete: %d flows, %d links, %d decisions in %.2fs",
            len(flows), len(links), len(decisions), iteration_time,
        )

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    def run(self) -> None:
        """Run the orchestrator loop."""
        logger.info("Starting orchestrator")
        logger.info("Poll interval: %ds", POLL_INTERVAL)
        logger.info(
            "ML classifier: %s",
            "enabled" if self.policy_engine.ml_classifier is not None else "disabled",
        )

        self.start_captures()

        while self.running:
            try:
                self.run_once()
            except Exception as e:
                logger.error("Unhandled error in orchestrator loop: %s", e, exc_info=True)
                SDN_PIPELINE_ITERATIONS.labels(status="error").inc()
            time.sleep(POLL_INTERVAL)

    def stop(self) -> None:
        """Gracefully stop the orchestrator."""
        self.running = False
        self.stop_captures()
        SDN_UP.set(0)
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
