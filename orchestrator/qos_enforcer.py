"""
QoS Enforcer Module for SDN Traffic Management.

Manages OVS flow rules for QoS enforcement including:
- Queue assignment (set_queue action)
- Flow rule installation/deletion
- Path rerouting for congestion avoidance

Uses ovs-ofctl for OpenFlow rule management and ovs-vsctl for QoS configuration.
"""

import subprocess
import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PathChoice(Enum):
    """Available network paths in the topology."""

    PATH_A = ("A", 3, "s2-s3")  # Via s1-eth3 -> s2 -> s3 -> s4-eth3
    PATH_B = ("B", 4, "s5-s6")  # Via s1-eth4 -> s5 -> s6 -> s4-eth4

    def __init__(self, label: str, egress_port: int, description: str):
        self.label = label
        self.egress_port = egress_port
        self.description = description


@dataclass
class FlowMatch:
    """Represents a flow matching criteria for OpenFlow rules."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # 6=TCP, 17=UDP

    @property
    def protocol_name(self) -> str:
        return "tcp" if self.protocol == 6 else "udp"

    @property
    def flow_id(self) -> str:
        """Generate a unique flow ID for tracking."""
        key = f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}/{self.protocol}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def to_ovs_match(self) -> str:
        """Convert to OVS flow match string."""
        parts = [
            f"{self.protocol_name}",
            f"nw_src={self.src_ip}",
            f"nw_dst={self.dst_ip}",
        ]

        # Add port matching for TCP/UDP
        if self.src_port > 0:
            port_field = "tp_src" if self.protocol in (6, 17) else "tp_src"
            parts.append(f"{port_field}={self.src_port}")
        if self.dst_port > 0:
            port_field = "tp_dst" if self.protocol in (6, 17) else "tp_dst"
            parts.append(f"{port_field}={self.dst_port}")

        return ",".join(parts)

    def to_reverse_match(self) -> "FlowMatch":
        """Get the reverse direction match for bidirectional flows."""
        return FlowMatch(
            src_ip=self.dst_ip,
            dst_ip=self.src_ip,
            src_port=self.dst_port,
            dst_port=self.src_port,
            protocol=self.protocol,
        )


@dataclass
class InstalledRule:
    """Tracks an installed OpenFlow rule."""

    switch: str
    match: FlowMatch
    queue_id: int
    priority: int
    cookie: int
    installed_at: float
    output_port: Optional[int] = None
    is_rerouted: bool = False


class QoSEnforcer:
    """
    Manages QoS enforcement through OVS OpenFlow rules.

    Responsibilities:
    - Install flow rules with set_queue action for priority queuing
    - Delete stale flow rules
    - Reroute flows to alternate paths
    - Track installed rules to avoid duplicates

    Usage:
        enforcer = QoSEnforcer()
        enforcer.install_qos_rule("s1", match, queue_id=0)  # P3 traffic
        enforcer.reroute_flow("s1", match, PathChoice.PATH_B)
    """

    # OpenFlow priority levels for our rules
    # Higher priority = matched first
    PRIORITY_QOS = 200  # Our QoS rules
    PRIORITY_REROUTE = 210  # Reroute rules (higher than normal QoS)

    # Cookie prefix for our rules (helps identify them)
    COOKIE_PREFIX = 0x5DC0  # "SDC" for SDN Controller

    # OpenFlow version
    OF_VERSION = "OpenFlow13"

    def __init__(self, dry_run: bool = False):
        """
        Initialize the QoS Enforcer.

        Args:
            dry_run: If True, log commands without executing them.
        """
        self.dry_run = dry_run
        self.installed_rules: Dict[str, InstalledRule] = {}
        self._cookie_counter = 1

        # Statistics
        self.stats = {
            "rules_installed": 0,
            "rules_deleted": 0,
            "reroutes": 0,
            "errors": 0,
        }

    def _run_ovs_cmd(self, cmd: str, check: bool = True) -> Tuple[bool, str]:
        """
        Execute an OVS command.

        Args:
            cmd: Command to execute
            check: If True, raise on error

        Returns:
            Tuple of (success, output)
        """
        if self.dry_run:
            logger.info(f"[DRY-RUN] {cmd}")
            return True, ""

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                if check:
                    logger.error(f"OVS command failed: {cmd}")
                    logger.error(f"stderr: {result.stderr}")
                    self.stats["errors"] += 1
                return False, result.stderr

            return True, result.stdout

        except subprocess.TimeoutExpired:
            logger.error(f"OVS command timed out: {cmd}")
            self.stats["errors"] += 1
            return False, "timeout"
        except Exception as e:
            logger.error(f"OVS command error: {cmd}, {e}")
            self.stats["errors"] += 1
            return False, str(e)

    def _generate_cookie(self) -> int:
        """Generate a unique cookie for flow tracking."""
        cookie = (self.COOKIE_PREFIX << 16) | (self._cookie_counter & 0xFFFF)
        self._cookie_counter += 1
        return cookie

    def _get_rule_key(self, switch: str, match: FlowMatch) -> str:
        """Generate a unique key for rule tracking."""
        return f"{switch}:{match.flow_id}"

    def install_qos_rule(
        self,
        switch: str,
        match: FlowMatch,
        queue_id: int,
        output_port: Optional[int] = None,
        bidirectional: bool = True,
    ) -> bool:
        """
        Install a QoS flow rule with set_queue action.

        Args:
            switch: Switch name (e.g., "s1")
            match: Flow matching criteria
            queue_id: Target queue ID (0-3)
            output_port: Specific output port (None for normal switching)
            bidirectional: If True, also install reverse direction rule

        Returns:
            True if successful
        """
        rule_key = self._get_rule_key(switch, match)

        # Check if rule already exists with same queue
        if rule_key in self.installed_rules:
            existing = self.installed_rules[rule_key]
            if existing.queue_id == queue_id and existing.output_port == output_port:
                logger.debug(f"Rule already exists: {rule_key}")
                return True
            else:
                # Queue or port changed, delete old rule first
                self.delete_rule(switch, match)

        cookie = self._generate_cookie()

        # Build actions
        if output_port is not None:
            actions = f"set_queue:{queue_id},output:{output_port}"
        else:
            actions = f"set_queue:{queue_id},normal"

        # Build ovs-ofctl command
        cmd = (
            f"ovs-ofctl add-flow {switch} -O {self.OF_VERSION} "
            f'"priority={self.PRIORITY_QOS},'
            f"cookie={cookie},"
            f"{match.to_ovs_match()},"
            f'actions={actions}"'
        )

        success, _ = self._run_ovs_cmd(cmd)

        if success:
            # Track the installed rule
            self.installed_rules[rule_key] = InstalledRule(
                switch=switch,
                match=match,
                queue_id=queue_id,
                priority=self.PRIORITY_QOS,
                cookie=cookie,
                installed_at=time.time(),
                output_port=output_port,
                is_rerouted=(output_port is not None),
            )
            self.stats["rules_installed"] += 1
            logger.info(
                f"Installed QoS rule: {switch} {match.src_ip}->{match.dst_ip} queue={queue_id}"
            )

            # Install reverse direction if requested
            if bidirectional:
                reverse_match = match.to_reverse_match()
                self.install_qos_rule(
                    switch,
                    reverse_match,
                    queue_id,
                    output_port=output_port,
                    bidirectional=False,  # Prevent infinite recursion
                )

        return success

    def delete_rule(
        self, switch: str, match: FlowMatch, bidirectional: bool = True
    ) -> bool:
        """
        Delete a flow rule.

        Args:
            switch: Switch name
            match: Flow matching criteria
            bidirectional: If True, also delete reverse direction rule

        Returns:
            True if successful
        """
        rule_key = self._get_rule_key(switch, match)

        # Get cookie if we have it
        cookie_match = ""
        if rule_key in self.installed_rules:
            cookie = self.installed_rules[rule_key].cookie
            cookie_match = f"cookie={cookie}/-1,"

        cmd = (
            f"ovs-ofctl del-flows {switch} -O {self.OF_VERSION} "
            f'"{cookie_match}{match.to_ovs_match()}"'
        )

        success, _ = self._run_ovs_cmd(cmd, check=False)

        if success:
            if rule_key in self.installed_rules:
                del self.installed_rules[rule_key]
            self.stats["rules_deleted"] += 1
            logger.info(f"Deleted rule: {switch} {match.src_ip}->{match.dst_ip}")

            # Delete reverse direction if requested
            if bidirectional:
                reverse_match = match.to_reverse_match()
                self.delete_rule(switch, reverse_match, bidirectional=False)

        return success

    def reroute_flow(
        self, switch: str, match: FlowMatch, path: PathChoice, queue_id: int
    ) -> bool:
        """
        Reroute a flow to an alternate path.

        Args:
            switch: Edge switch name (e.g., "s1")
            match: Flow matching criteria
            path: Target path (PATH_A or PATH_B)
            queue_id: Queue ID to use on the rerouted path

        Returns:
            True if successful
        """
        logger.info(
            f"Rerouting flow {match.src_ip}->{match.dst_ip} to {path.description}"
        )

        # Use higher priority for reroute rules
        rule_key = self._get_rule_key(switch, match)

        # Delete existing rule if any
        if rule_key in self.installed_rules:
            self.delete_rule(switch, match, bidirectional=False)

        cookie = self._generate_cookie()

        # Build reroute rule with specific output port
        cmd = (
            f"ovs-ofctl add-flow {switch} -O {self.OF_VERSION} "
            f'"priority={self.PRIORITY_REROUTE},'
            f"cookie={cookie},"
            f"{match.to_ovs_match()},"
            f'actions=set_queue:{queue_id},output:{path.egress_port}"'
        )

        success, _ = self._run_ovs_cmd(cmd)

        if success:
            self.installed_rules[rule_key] = InstalledRule(
                switch=switch,
                match=match,
                queue_id=queue_id,
                priority=self.PRIORITY_REROUTE,
                cookie=cookie,
                installed_at=time.time(),
                output_port=path.egress_port,
                is_rerouted=True,
            )
            self.stats["reroutes"] += 1

            # Also install reverse direction on the other edge switch
            # For s1 flows, the reverse comes back through s4
            reverse_switch = "s4" if switch == "s1" else "s1"
            reverse_match = match.to_reverse_match()

            # Determine return path output port
            return_port = 3 if path == PathChoice.PATH_A else 4

            self.install_qos_rule(
                reverse_switch,
                reverse_match,
                queue_id,
                output_port=return_port,
                bidirectional=False,
            )

        return success

    def clear_reroute(self, switch: str, match: FlowMatch) -> bool:
        """
        Clear reroute and restore normal forwarding.

        Args:
            switch: Switch name
            match: Flow matching criteria

        Returns:
            True if successful
        """
        rule_key = self._get_rule_key(switch, match)

        if rule_key in self.installed_rules:
            rule = self.installed_rules[rule_key]
            if rule.is_rerouted:
                # Delete reroute rule
                self.delete_rule(switch, match)
                # Reinstall as normal QoS rule
                self.install_qos_rule(switch, match, rule.queue_id)
                logger.info(f"Cleared reroute for {match.src_ip}->{match.dst_ip}")
                return True

        return False

    def cleanup_stale_rules(self, max_age_seconds: float = 300) -> int:
        """
        Remove rules older than max_age that haven't been refreshed.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of rules removed
        """
        now = time.time()
        stale_keys = [
            key
            for key, rule in self.installed_rules.items()
            if (now - rule.installed_at) > max_age_seconds
        ]

        count = 0
        for key in stale_keys:
            rule = self.installed_rules[key]
            if self.delete_rule(rule.switch, rule.match, bidirectional=False):
                count += 1

        if count > 0:
            logger.info(f"Cleaned up {count} stale rules")

        return count

    def clear_all_rules(self, switch: Optional[str] = None) -> int:
        """
        Clear all rules installed by this enforcer.

        Args:
            switch: Specific switch to clear, or None for all

        Returns:
            Number of rules cleared
        """
        to_delete = []
        for key, rule in self.installed_rules.items():
            if switch is None or rule.switch == switch:
                to_delete.append((rule.switch, rule.match))

        count = 0
        for sw, match in to_delete:
            if self.delete_rule(sw, match, bidirectional=False):
                count += 1

        logger.info(f"Cleared {count} rules" + (f" on {switch}" if switch else ""))
        return count

    def get_installed_rules(self) -> List[Dict]:
        """Get list of installed rules as dictionaries."""
        return [
            {
                "switch": r.switch,
                "flow_id": m.flow_id,
                "src": f"{r.match.src_ip}:{r.match.src_port}",
                "dst": f"{r.match.dst_ip}:{r.match.dst_port}",
                "protocol": r.match.protocol_name,
                "queue_id": r.queue_id,
                "output_port": r.output_port,
                "is_rerouted": r.is_rerouted,
                "age": time.time() - r.installed_at,
            }
            for m, r in [
                (self.installed_rules[k].match, self.installed_rules[k])
                for k in self.installed_rules
            ]
        ]

    def get_stats(self) -> Dict:
        """Get enforcer statistics."""
        return {
            **self.stats,
            "active_rules": len(self.installed_rules),
            "rerouted_flows": sum(
                1 for r in self.installed_rules.values() if r.is_rerouted
            ),
        }


# Convenience function for creating FlowMatch from dict
def flow_match_from_dict(flow: Dict[str, Any]) -> FlowMatch:
    """
    Create a FlowMatch from a flow dictionary.

    Args:
        flow: Dictionary with src_ip, dst_ip, src_port, dst_port, protocol

    Returns:
        FlowMatch object
    """
    protocol = flow.get("protocol", 6)
    if isinstance(protocol, str):
        protocol = 17 if protocol.lower() == "udp" else 6

    return FlowMatch(
        src_ip=flow.get("src_ip", "0.0.0.0"),
        dst_ip=flow.get("dst_ip", "0.0.0.0"),
        src_port=int(flow.get("src_port", 0)),
        dst_port=int(flow.get("dst_port", 0)),
        protocol=int(protocol),
    )


if __name__ == "__main__":
    # Self-test
    logging.basicConfig(level=logging.INFO)

    print("QoS Enforcer Test (Dry Run)")
    print("=" * 60)

    enforcer = QoSEnforcer(dry_run=True)

    # Test flow match
    match = FlowMatch(
        src_ip="10.0.0.1", dst_ip="10.0.0.3", src_port=50000, dst_port=443, protocol=6
    )

    print(f"\nFlow Match: {match.to_ovs_match()}")
    print(f"Flow ID: {match.flow_id}")

    # Test QoS rule installation
    print("\n1. Installing QoS rule (Queue 0 - Banking)...")
    enforcer.install_qos_rule("s1", match, queue_id=0)

    # Test reroute
    print("\n2. Rerouting to Path B...")
    enforcer.reroute_flow("s1", match, PathChoice.PATH_B, queue_id=0)

    # Show stats
    print("\n3. Stats:")
    for key, value in enforcer.get_stats().items():
        print(f"  {key}: {value}")

    # Show installed rules
    print("\n4. Installed Rules:")
    for rule in enforcer.get_installed_rules():
        print(f"  {rule}")

    # Cleanup
    print("\n5. Clearing all rules...")
    enforcer.clear_all_rules()

    print("\nTest complete!")
