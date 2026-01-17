"""
OVS Statistics Scraper
Collects flow and port statistics from Open vSwitch.
"""

import subprocess
import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FlowStats:
    """Statistics for a single flow"""

    cookie: str
    duration: float
    table: int
    n_packets: int
    n_bytes: int
    priority: int
    match: str
    actions: str


@dataclass
class PortStats:
    """Statistics for a switch port"""

    port_no: int
    rx_packets: int
    tx_packets: int
    rx_bytes: int
    tx_bytes: int
    rx_dropped: int
    tx_dropped: int
    rx_errors: int
    tx_errors: int


class OVSScraper:
    """Scrapes statistics from OVS switches"""

    def __init__(self, bridge: str = "s1"):
        self.bridge = bridge

    def get_flow_stats(self) -> List[FlowStats]:
        """
        Get flow statistics from OVS.

        Returns:
            List of FlowStats objects
        """
        try:
            result = subprocess.run(
                ["ovs-ofctl", "dump-flows", self.bridge],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                logger.error(f"ovs-ofctl error: {result.stderr}")
                return []

            return self._parse_flow_stats(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("ovs-ofctl timed out")
            return []
        except FileNotFoundError:
            logger.error("ovs-ofctl not found")
            return []

    def _parse_flow_stats(self, output: str) -> List[FlowStats]:
        """Parse ovs-ofctl dump-flows output"""
        flows = []

        for line in output.strip().split("\n"):
            if (
                not line
                or line.startswith("NXST_FLOW")
                or line.startswith("OFPST_FLOW")
            ):
                continue

            try:
                # Parse flow entry
                # Example: cookie=0x0, duration=100.5s, table=0, n_packets=1000, n_bytes=50000, priority=1,in_port=1 actions=output:2

                cookie_match = re.search(r"cookie=(\w+)", line)
                duration_match = re.search(r"duration=([\d.]+)s", line)
                table_match = re.search(r"table=(\d+)", line)
                packets_match = re.search(r"n_packets=(\d+)", line)
                bytes_match = re.search(r"n_bytes=(\d+)", line)
                priority_match = re.search(r"priority=(\d+)", line)
                actions_match = re.search(r"actions=(.+)$", line)

                # Extract match criteria (everything between priority= and actions=)
                match_pattern = re.search(r"priority=\d+,?(.+?)\s*actions=", line)
                match_str = match_pattern.group(1).strip(",") if match_pattern else ""

                flow = FlowStats(
                    cookie=cookie_match.group(1) if cookie_match else "0x0",
                    duration=float(duration_match.group(1)) if duration_match else 0.0,
                    table=int(table_match.group(1)) if table_match else 0,
                    n_packets=int(packets_match.group(1)) if packets_match else 0,
                    n_bytes=int(bytes_match.group(1)) if bytes_match else 0,
                    priority=int(priority_match.group(1)) if priority_match else 0,
                    match=match_str,
                    actions=actions_match.group(1) if actions_match else "",
                )
                flows.append(flow)

            except Exception as e:
                logger.debug(f"Failed to parse flow line: {line}, error: {e}")

        return flows

    def get_port_stats(self) -> List[PortStats]:
        """
        Get port statistics from OVS.

        Returns:
            List of PortStats objects
        """
        try:
            result = subprocess.run(
                ["ovs-ofctl", "dump-ports", self.bridge],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                logger.error(f"ovs-ofctl error: {result.stderr}")
                return []

            return self._parse_port_stats(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("ovs-ofctl timed out")
            return []
        except FileNotFoundError:
            logger.error("ovs-ofctl not found")
            return []

    def _parse_port_stats(self, output: str) -> List[PortStats]:
        """Parse ovs-ofctl dump-ports output"""
        ports = []

        # Port stats come in multi-line format
        current_port = None

        for line in output.split("\n"):
            # Port number line
            port_match = re.search(r"port\s+(\d+|LOCAL):", line)
            if port_match:
                port_str = port_match.group(1)
                current_port = {
                    "port_no": -1 if port_str == "LOCAL" else int(port_str),
                    "rx_packets": 0,
                    "tx_packets": 0,
                    "rx_bytes": 0,
                    "tx_bytes": 0,
                    "rx_dropped": 0,
                    "tx_dropped": 0,
                    "rx_errors": 0,
                    "tx_errors": 0,
                }

                # Parse inline stats
                rx_match = re.search(
                    r"rx pkts=(\d+), bytes=(\d+), drop=(\d+), errs=(\d+)", line
                )
                tx_match = re.search(
                    r"tx pkts=(\d+), bytes=(\d+), drop=(\d+), errs=(\d+)", line
                )

                if rx_match:
                    current_port["rx_packets"] = int(rx_match.group(1))
                    current_port["rx_bytes"] = int(rx_match.group(2))
                    current_port["rx_dropped"] = int(rx_match.group(3))
                    current_port["rx_errors"] = int(rx_match.group(4))

                if tx_match:
                    current_port["tx_packets"] = int(tx_match.group(1))
                    current_port["tx_bytes"] = int(tx_match.group(2))
                    current_port["tx_dropped"] = int(tx_match.group(3))
                    current_port["tx_errors"] = int(tx_match.group(4))

                if current_port["port_no"] != -1:  # Skip LOCAL port
                    ports.append(PortStats(**current_port))

        return ports

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of bridge statistics"""
        flows = self.get_flow_stats()
        ports = self.get_port_stats()

        total_bytes = sum(p.rx_bytes + p.tx_bytes for p in ports)
        total_packets = sum(p.rx_packets + p.tx_packets for p in ports)

        return {
            "timestamp": datetime.now().isoformat(),
            "bridge": self.bridge,
            "num_flows": len(flows),
            "num_ports": len(ports),
            "total_bytes": total_bytes,
            "total_packets": total_packets,
            "flows": [f.__dict__ for f in flows],
            "ports": [p.__dict__ for p in ports],
        }


def collect_stats(bridge: str = "s1") -> Dict[str, Any]:
    """Convenience function to collect OVS stats"""
    scraper = OVSScraper(bridge)
    return scraper.get_summary()


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.DEBUG)

    print("OVS Scraper Test")
    print("-" * 50)

    scraper = OVSScraper("s1")

    print("\nFlow Stats:")
    flows = scraper.get_flow_stats()
    for f in flows:
        print(f"  {f}")

    print("\nPort Stats:")
    ports = scraper.get_port_stats()
    for p in ports:
        print(f"  {p}")

    print("\nSummary:")
    summary = scraper.get_summary()
    print(json.dumps(summary, indent=2, default=str))
