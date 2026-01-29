"""
Packet Capture and Feature Extraction Module using Scapy.
Extracts statistical flow features and TLS SNI for ML traffic classification.
"""

import threading
import time
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from scapy.all import sniff, IP, TCP, UDP, load_layer, AsyncSniffer
from scapy.layers.tls.all import TLS, TLSClientHello
from scapy.packet import Packet

# Ensure TLS layer is loaded
load_layer("tls")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FlowStats:
    """Accumulated statistics for a bidirectional network flow."""

    # 5-tuple identity
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # 6=TCP, 17=UDP

    # Timestamps
    start_time: float
    last_seen: float

    # Counters
    packet_count: int = 0
    byte_count: int = 0

    # Payload sizes (for stats)
    packet_sizes: List[int] = field(default_factory=list)

    # Inter-arrival times
    last_packet_time: float = 0.0
    iat_list: List[float] = field(default_factory=list)

    # Deep Packet Inspection
    sni: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.last_seen - self.start_time

    @property
    def flow_key(self) -> Tuple:
        """Canonical key for bidirectional flow."""
        if self.src_ip < self.dst_ip:
            return (
                self.src_ip,
                self.dst_ip,
                self.src_port,
                self.dst_port,
                self.protocol,
            )
        else:
            return (
                self.dst_ip,
                self.src_ip,
                self.dst_port,
                self.src_port,
                self.protocol,
            )

    def update(self, packet_len: int, timestamp: float, sni: Optional[str] = None):
        """Update flow stats with a new packet."""
        self.packet_count += 1
        self.byte_count += packet_len
        self.packet_sizes.append(packet_len)

        # Only calc IAT if it's not the very first packet

        # (Or first packet of the *session* for this object instance)
        # We initialize last_packet_time = timestamp on creation, so first update
        # might yield IAT=0 if created and updated same time, or delta if created earlier?
        # Actually in _process_packet we create then update.
        # So IAT will be 0 on first packet. We should ignore that.

        iat = timestamp - self.last_packet_time
        if iat > 0.000001:  # Ignore self-update or identical timestamps
            self.iat_list.append(iat)

        self.last_packet_time = timestamp
        self.last_seen = timestamp

        if sni and not self.sni:
            self.sni = sni

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML input/DataFrame."""
        sizes = self.packet_sizes
        iats = self.iat_list

        return {
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "packet_count": self.packet_count,
            "byte_count": self.byte_count,
            "duration": self.duration,
            "pkt_len_min": min(sizes) if sizes else 0,
            "pkt_len_max": max(sizes) if sizes else 0,
            "pkt_len_mean": statistics.mean(sizes) if sizes else 0,
            "pkt_len_std": statistics.stdev(sizes) if len(sizes) > 1 else 0,
            "iat_mean": statistics.mean(iats) if iats else 0,
            "iat_std": statistics.stdev(iats) if len(iats) > 1 else 0,
            "sni": self.sni if self.sni else "unknown",
        }


class PacketCapture:
    """
    Captures packets from an interface and aggregates them into flows.
    Uses AsyncSniffer for cleaner background execution.
    """

    def __init__(self, interface: str = "eth0", idle_timeout: int = 30):
        self.interface = interface
        self.idle_timeout = idle_timeout
        self.sniffer: Optional[AsyncSniffer] = None
        self.prune_timer: Optional[threading.Timer] = None

        # Flow storage: key -> FlowStats
        self.flows: Dict[Tuple, FlowStats] = {}
        self.lock = threading.Lock()

    def start(self):
        """Start the packet capture."""
        if self.sniffer and self.sniffer.running:
            logger.warning("Packet capture already running.")
            return

        logger.info(f"Starting AsyncSniffer on {self.interface}...")

        self.sniffer = AsyncSniffer(
            iface=self.interface,
            prn=self._process_packet,
            store=False,
            filter="ip",  # Capture IP traffic only
        )
        self.sniffer.start()

        # Start pruning loop
        self._schedule_prune()

    def stop(self):
        """Stop the packet capture."""
        logger.info("Stopping packet capture...")
        if self.sniffer:
            self.sniffer.stop()
            self.sniffer = None

        if self.prune_timer:
            self.prune_timer.cancel()
            self.prune_timer = None

    def _schedule_prune(self):
        """Schedule the next prune cycle."""
        self.prune_timer = threading.Timer(1.0, self._prune_flows_loop)
        self.prune_timer.daemon = True
        self.prune_timer.start()

    def _prune_flows_loop(self):
        """Timer callback for pruning."""
        self._prune_flows()
        # Reschedule if still running (checked via sniffer existence)
        if self.sniffer and self.sniffer.running:
            self._schedule_prune()

    def process_pcap(self, pcap_path: str):
        """Process packets from a pcap file (offline mode)."""
        logger.info(f"Processing pcap file: {pcap_path}")
        try:
            sniff(offline=pcap_path, prn=self._process_packet, store=False)
        except Exception as e:
            logger.error(f"Error processing pcap {pcap_path}: {e}")

    def _process_packet(self, pkt: Packet):
        """Callback for each captured packet."""
        if not pkt.haslayer(IP):
            return

        timestamp = float(pkt.time)
        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto
        length = len(pkt)

        src_port = 0
        dst_port = 0
        sni = None

        # Extract Ports and SNI
        if pkt.haslayer(TCP):
            tcp_layer = pkt[TCP]
            src_port = int(tcp_layer.sport)
            dst_port = int(tcp_layer.dport)

            # Try extraction of TLS SNI
            if pkt.haslayer(TLSClientHello):
                try:
                    client_hello = pkt[TLSClientHello]
                    if hasattr(client_hello, "ext"):
                        for ext in client_hello.ext:
                            if ext.type == 0:  # server_name
                                if (
                                    hasattr(ext, "servernames")
                                    and len(ext.servernames) > 0
                                ):
                                    sni_bytes = ext.servernames[0].servername
                                    sni = sni_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    pass

        elif pkt.haslayer(UDP):
            udp_layer = pkt[UDP]
            src_port = int(udp_layer.sport)
            dst_port = int(udp_layer.dport)

        # Construct Canonical Key
        if src_ip < dst_ip:
            key = (src_ip, dst_ip, src_port, dst_port, proto)
            c_src, c_dst = src_ip, dst_ip
            c_sport, c_dport = src_port, dst_port
        else:
            key = (dst_ip, src_ip, dst_port, src_port, proto)
            c_src, c_dst = dst_ip, src_ip
            c_sport, c_dport = dst_port, src_port

        with self.lock:
            if key not in self.flows:
                self.flows[key] = FlowStats(
                    src_ip=c_src,
                    dst_ip=c_dst,
                    src_port=c_sport,
                    dst_port=c_dport,
                    protocol=proto,
                    start_time=timestamp,
                    last_seen=timestamp,
                    last_packet_time=timestamp,
                )

            self.flows[key].update(length, timestamp, sni)

    def _prune_flows(self):
        """Remove inactive flows."""
        now = time.time()
        with self.lock:
            # Create list of keys to remove
            keys_to_remove = [
                k
                for k, v in self.flows.items()
                if (now - v.last_seen) > self.idle_timeout
            ]
            for k in keys_to_remove:
                del self.flows[k]

    def get_flow_stats(self) -> List[Dict[str, Any]]:
        """Return a snapshot of current flow statistics."""
        with self.lock:
            return [flow.to_dict() for flow in self.flows.values()]
