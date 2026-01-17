"""
Prometheus Metrics Scraper
Collects metrics from Faucet's Prometheus exporter.
"""

import requests
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class PrometheusMetric:
    """A single Prometheus metric"""

    name: str
    labels: Dict[str, str]
    value: float
    timestamp: Optional[float] = None


class PrometheusScraper:
    """Scrapes metrics from Prometheus server"""

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url.rstrip("/")
        self.session = requests.Session()

    def query(self, query: str) -> List[Dict]:
        """
        Execute a PromQL query.

        Args:
            query: PromQL query string

        Returns:
            List of result dictionaries
        """
        try:
            response = self.session.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=5,
            )
            response.raise_for_status()

            data = response.json()
            if data["status"] != "success":
                logger.error(f"Prometheus query failed: {data}")
                return []

            return data.get("data", {}).get("result", [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Prometheus request failed: {e}")
            return []

    def query_range(
        self, query: str, start: str, end: str, step: str = "10s"
    ) -> List[Dict]:
        """
        Execute a range query.

        Args:
            query: PromQL query string
            start: Start time (RFC3339 or Unix timestamp)
            end: End time
            step: Query resolution step

        Returns:
            List of result dictionaries with values over time
        """
        try:
            response = self.session.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start,
                    "end": end,
                    "step": step,
                },
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            if data["status"] != "success":
                logger.error(f"Prometheus query failed: {data}")
                return []

            return data.get("data", {}).get("result", [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Prometheus request failed: {e}")
            return []

    def get_faucet_port_stats(self) -> List[Dict]:
        """Get Faucet port statistics"""
        metrics = {}

        # Query port bytes
        for direction in ["rx", "tx"]:
            results = self.query(f"faucet_port_{direction}_bytes")
            for result in results:
                labels = result.get("metric", {})
                dp = labels.get("dp_name", "unknown")
                port = labels.get("port", "unknown")
                key = f"{dp}:{port}"

                if key not in metrics:
                    metrics[key] = {
                        "dp_name": dp,
                        "port": port,
                        "timestamp": datetime.now().isoformat(),
                    }

                value = float(result.get("value", [0, 0])[1])
                metrics[key][f"{direction}_bytes"] = value

        # Query port packets
        for direction in ["rx", "tx"]:
            results = self.query(f"faucet_port_{direction}_packets")
            for result in results:
                labels = result.get("metric", {})
                dp = labels.get("dp_name", "unknown")
                port = labels.get("port", "unknown")
                key = f"{dp}:{port}"

                if key in metrics:
                    value = float(result.get("value", [0, 0])[1])
                    metrics[key][f"{direction}_packets"] = value

        return list(metrics.values())

    def get_faucet_flow_stats(self) -> List[Dict]:
        """Get Faucet flow statistics"""
        results = self.query("faucet_flow_packet_count")

        flows = []
        for result in results:
            labels = result.get("metric", {})
            value = float(result.get("value", [0, 0])[1])

            flows.append(
                {
                    "dp_name": labels.get("dp_name", "unknown"),
                    "table_id": labels.get("table_id", "0"),
                    "cookie": labels.get("cookie", "0"),
                    "packet_count": value,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return flows

    def get_link_utilization(self, capacity_bps: int = 10_000_000) -> List[Dict]:
        """
        Calculate link utilization based on byte rates.

        Args:
            capacity_bps: Link capacity in bits per second

        Returns:
            List of utilization metrics per port
        """
        # Query rate of change in bytes over last minute
        results = self.query(
            "rate(faucet_port_rx_bytes[1m]) + rate(faucet_port_tx_bytes[1m])"
        )

        utilization = []
        for result in results:
            labels = result.get("metric", {})
            bytes_per_sec = float(result.get("value", [0, 0])[1])
            bits_per_sec = bytes_per_sec * 8
            util = min(1.0, bits_per_sec / capacity_bps)

            utilization.append(
                {
                    "dp_name": labels.get("dp_name", "unknown"),
                    "port": labels.get("port", "unknown"),
                    "bytes_per_sec": bytes_per_sec,
                    "bits_per_sec": bits_per_sec,
                    "utilization": util,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return utilization

    def health_check(self) -> bool:
        """Check if Prometheus is healthy"""
        try:
            response = self.session.get(f"{self.prometheus_url}/-/healthy", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


def collect_faucet_metrics(
    prometheus_url: str = "http://localhost:9090",
) -> Dict[str, Any]:
    """Convenience function to collect all Faucet metrics"""
    scraper = PrometheusScraper(prometheus_url)

    return {
        "timestamp": datetime.now().isoformat(),
        "healthy": scraper.health_check(),
        "port_stats": scraper.get_faucet_port_stats(),
        "flow_stats": scraper.get_faucet_flow_stats(),
        "utilization": scraper.get_link_utilization(),
    }


if __name__ == "__main__":
    import json
    import os

    logging.basicConfig(level=logging.INFO)

    prometheus_url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")

    print(f"Prometheus Scraper Test (URL: {prometheus_url})")
    print("-" * 50)

    scraper = PrometheusScraper(prometheus_url)

    print(f"\nHealth check: {scraper.health_check()}")

    print("\nPort Stats:")
    port_stats = scraper.get_faucet_port_stats()
    print(json.dumps(port_stats, indent=2))

    print("\nFlow Stats:")
    flow_stats = scraper.get_faucet_flow_stats()
    print(json.dumps(flow_stats, indent=2))

    print("\nLink Utilization:")
    util = scraper.get_link_utilization()
    print(json.dumps(util, indent=2))
