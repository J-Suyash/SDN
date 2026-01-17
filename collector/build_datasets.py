"""
Dataset Builder
Transforms raw telemetry into ML-ready datasets.
"""

import os
import csv
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class FlowRecord:
    """A single flow record for the classifier dataset"""

    flow_id: str
    timestamp: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packet_count: int
    byte_count: int
    duration_sec: float
    bytes_per_packet: float
    packets_per_sec: float
    bytes_per_sec: float
    sni_domain: str
    label: str  # P0, P1, P2, P3


@dataclass
class LinkRecord:
    """A single link record for the predictor dataset"""

    timestamp: str
    switch: str
    port: int
    bytes_delta: int
    utilization: float
    hour_of_day: int
    minute_of_hour: int
    is_weekday: bool
    label: int  # 1 if congested in next interval, 0 otherwise


class DatasetBuilder:
    """Builds ML training datasets from telemetry data"""

    def __init__(self, data_dir: str = "/app/data/processed"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.flows_file = os.path.join(data_dir, "flows.csv")
        self.links_file = os.path.join(data_dir, "link_timeseries.csv")

        # Initialize CSV files with headers if they don't exist
        self._init_csv_files()

    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        if not os.path.exists(self.flows_file):
            with open(self.flows_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(FlowRecord.__dataclass_fields__.keys())
                )
                writer.writeheader()

        if not os.path.exists(self.links_file):
            with open(self.links_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(LinkRecord.__dataclass_fields__.keys())
                )
                writer.writeheader()

    def add_flow(self, flow: FlowRecord) -> None:
        """Add a flow record to the dataset"""
        with open(self.flows_file, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(FlowRecord.__dataclass_fields__.keys())
            )
            writer.writerow(asdict(flow))

    def add_flows(self, flows: List[FlowRecord]) -> None:
        """Add multiple flow records"""
        with open(self.flows_file, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(FlowRecord.__dataclass_fields__.keys())
            )
            for flow in flows:
                writer.writerow(asdict(flow))

    def add_link_record(self, record: LinkRecord) -> None:
        """Add a link record to the dataset"""
        with open(self.links_file, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(LinkRecord.__dataclass_fields__.keys())
            )
            writer.writerow(asdict(record))

    def add_link_records(self, records: List[LinkRecord]) -> None:
        """Add multiple link records"""
        with open(self.links_file, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(LinkRecord.__dataclass_fields__.keys())
            )
            for record in records:
                writer.writerow(asdict(record))

    def build_flow_record(
        self, raw_flow: Dict[str, Any], label: str, sni_domain: str = ""
    ) -> FlowRecord:
        """
        Build a FlowRecord from raw flow data.

        Args:
            raw_flow: Dictionary with flow information
            label: Priority label (P0, P1, P2, P3)
            sni_domain: SNI domain if captured

        Returns:
            FlowRecord object
        """
        packet_count = raw_flow.get("packet_count", raw_flow.get("n_packets", 0))
        byte_count = raw_flow.get("byte_count", raw_flow.get("n_bytes", 0))
        duration = raw_flow.get("duration", raw_flow.get("duration_sec", 1))

        # Avoid division by zero
        duration = max(duration, 0.001)
        packet_count = max(packet_count, 1)

        return FlowRecord(
            flow_id=raw_flow.get("flow_id", raw_flow.get("cookie", "unknown")),
            timestamp=raw_flow.get("timestamp", datetime.now().isoformat()),
            src_ip=raw_flow.get("src_ip", ""),
            dst_ip=raw_flow.get("dst_ip", ""),
            src_port=raw_flow.get("src_port", 0),
            dst_port=raw_flow.get("dst_port", 0),
            protocol=raw_flow.get("protocol", "tcp"),
            packet_count=packet_count,
            byte_count=byte_count,
            duration_sec=duration,
            bytes_per_packet=byte_count / packet_count,
            packets_per_sec=packet_count / duration,
            bytes_per_sec=byte_count / duration,
            sni_domain=sni_domain,
            label=label,
        )

    def build_link_record(
        self,
        switch: str,
        port: int,
        bytes_delta: int,
        capacity_bps: int = 10_000_000,
        interval_sec: int = 10,
        next_congested: bool = False,
    ) -> LinkRecord:
        """
        Build a LinkRecord from raw telemetry.

        Args:
            switch: Switch identifier
            port: Port number
            bytes_delta: Bytes transferred in this interval
            capacity_bps: Link capacity in bits per second
            interval_sec: Collection interval in seconds
            next_congested: Whether link is congested in next interval (label)

        Returns:
            LinkRecord object
        """
        now = datetime.now()
        bits_delta = bytes_delta * 8
        utilization = bits_delta / (capacity_bps * interval_sec)
        utilization = min(1.0, max(0.0, utilization))

        return LinkRecord(
            timestamp=now.isoformat(),
            switch=switch,
            port=port,
            bytes_delta=bytes_delta,
            utilization=utilization,
            hour_of_day=now.hour,
            minute_of_hour=now.minute,
            is_weekday=now.weekday() < 5,
            label=1 if next_congested else 0,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        flow_count = 0
        link_count = 0

        if os.path.exists(self.flows_file):
            with open(self.flows_file, "r") as f:
                flow_count = sum(1 for _ in f) - 1  # Exclude header

        if os.path.exists(self.links_file):
            with open(self.links_file, "r") as f:
                link_count = sum(1 for _ in f) - 1  # Exclude header

        return {
            "flows_file": self.flows_file,
            "links_file": self.links_file,
            "flow_records": max(0, flow_count),
            "link_records": max(0, link_count),
        }


def label_flow_by_port(dst_port: int) -> str:
    """
    Label a flow based on destination port.

    MVP labeling strategy - will be replaced with SNI-based labeling.
    """
    # Banking ports
    if dst_port in [443, 5003]:
        return "P3"

    # Voice ports
    if dst_port in [5060, 5061, 5002]:
        return "P2"

    # Bulk ports
    if dst_port in [20, 21, 22, 5000]:
        return "P0"

    # Default to web
    return "P1"


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    # Test with temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Dataset Builder Test")
        print("-" * 50)

        builder = DatasetBuilder(tmpdir)

        # Add some test flows
        test_flows = [
            {
                "flow_id": "1",
                "dst_port": 443,
                "packet_count": 100,
                "byte_count": 50000,
                "duration": 10,
            },
            {
                "flow_id": "2",
                "dst_port": 5002,
                "packet_count": 1000,
                "byte_count": 80000,
                "duration": 60,
            },
            {
                "flow_id": "3",
                "dst_port": 80,
                "packet_count": 50,
                "byte_count": 25000,
                "duration": 5,
            },
        ]

        for flow in test_flows:
            label = label_flow_by_port(flow["dst_port"])
            record = builder.build_flow_record(flow, label)
            builder.add_flow(record)
            print(f"Added flow: {record.flow_id} -> {label}")

        # Add some test link records
        for i in range(5):
            record = builder.build_link_record(
                switch="s1",
                port=1,
                bytes_delta=500000 + i * 100000,
                next_congested=(i > 3),
            )
            builder.add_link_record(record)
            print(f"Added link record: util={record.utilization:.2%}")

        # Print stats
        print("\nDataset Stats:")
        print(json.dumps(builder.get_stats(), indent=2))

        # Show file contents
        print("\nFlows CSV:")
        with open(builder.flows_file, "r") as f:
            print(f.read())

        print("Links CSV:")
        with open(builder.links_file, "r") as f:
            print(f.read())
