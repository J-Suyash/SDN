import subprocess
import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PathChoice(Enum):
    PATH_A = ("A", 3, "s2-s3")  # Via s1-eth3 -> s2 -> s3 -> s4-eth3
    PATH_B = ("B", 4, "s5-s6")  # Via s1-eth4 -> s5 -> s6 -> s4-eth4

    def __init__(self, label: str, egress_port: int, description: str):
        self.label = label
        self.egress_port = egress_port
        self.description = description


@dataclass
class FlowMatch:
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
        key = f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}/{self.protocol}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def to_ovs_match(self) -> str:
        parts = [
            self.protocol_name,
            f"nw_src={self.src_ip}",
            f"nw_dst={self.dst_ip}",
        ]
        if self.src_port > 0:
            parts.append(f"tp_src={self.src_port}")
        if self.dst_port > 0:
            parts.append(f"tp_dst={self.dst_port}")
        return ",".join(parts)

    def to_reverse_match(self) -> "FlowMatch":
        return FlowMatch(
            src_ip=self.dst_ip, dst_ip=self.src_ip,
            src_port=self.dst_port, dst_port=self.src_port,
            protocol=self.protocol,
        )


@dataclass
class InstalledRule:
    switch: str
    match: FlowMatch
    queue_id: int
    priority: int
    cookie: int
    installed_at: float
    output_port: Optional[int] = None
    is_rerouted: bool = False


class QoSEnforcer:
    PRIORITY_QOS = 200
    PRIORITY_REROUTE = 210
    COOKIE_PREFIX = 0x5DC0
    OF_VERSION = "OpenFlow13"

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.installed_rules: Dict[str, InstalledRule] = {}
        self._cookie_counter = 1
        self.stats = {
            "rules_installed": 0,
            "rules_deleted": 0,
            "reroutes": 0,
            "errors": 0,
        }

    def _run_ovs_cmd(self, cmd: str, check: bool = True) -> Tuple[bool, str]:
        if self.dry_run:
            logger.info(f"[DRY-RUN] {cmd}")
            return True, ""

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                if check:
                    logger.error(f"OVS command failed: {cmd}\nstderr: {result.stderr}")
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
        cookie = (self.COOKIE_PREFIX << 16) | (self._cookie_counter & 0xFFFF)
        self._cookie_counter += 1
        return cookie

    def _get_rule_key(self, switch: str, match: FlowMatch) -> str:
        return f"{switch}:{match.flow_id}"

    def install_qos_rule(
        self, switch: str, match: FlowMatch, queue_id: int,
        output_port: Optional[int] = None, bidirectional: bool = True,
    ) -> bool:
        rule_key = self._get_rule_key(switch, match)

        if rule_key in self.installed_rules:
            existing = self.installed_rules[rule_key]
            if existing.queue_id == queue_id and existing.output_port == output_port:
                return True
            self.delete_rule(switch, match)

        cookie = self._generate_cookie()

        if output_port is not None:
            actions = f"set_queue:{queue_id},output:{output_port}"
        else:
            actions = f"set_queue:{queue_id},normal"

        cmd = (
            f"ovs-ofctl add-flow {switch} -O {self.OF_VERSION} "
            f'"priority={self.PRIORITY_QOS},'
            f"cookie={cookie},"
            f"{match.to_ovs_match()},"
            f'actions={actions}"'
        )

        success, _ = self._run_ovs_cmd(cmd)

        if success:
            self.installed_rules[rule_key] = InstalledRule(
                switch=switch, match=match, queue_id=queue_id,
                priority=self.PRIORITY_QOS, cookie=cookie,
                installed_at=time.time(), output_port=output_port,
                is_rerouted=(output_port is not None),
            )
            self.stats["rules_installed"] += 1
            logger.info(f"Installed QoS rule: {switch} {match.src_ip}->{match.dst_ip} queue={queue_id}")

            if bidirectional:
                self.install_qos_rule(
                    switch, match.to_reverse_match(), queue_id,
                    output_port=output_port, bidirectional=False,
                )

        return success

    def delete_rule(self, switch: str, match: FlowMatch, bidirectional: bool = True) -> bool:
        rule_key = self._get_rule_key(switch, match)

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
            self.installed_rules.pop(rule_key, None)
            self.stats["rules_deleted"] += 1

            if bidirectional:
                self.delete_rule(switch, match.to_reverse_match(), bidirectional=False)

        return success

    def reroute_flow(
        self, switch: str, match: FlowMatch, path: PathChoice, queue_id: int,
    ) -> bool:
        logger.info(f"Rerouting flow {match.src_ip}->{match.dst_ip} to {path.description}")

        rule_key = self._get_rule_key(switch, match)
        if rule_key in self.installed_rules:
            self.delete_rule(switch, match, bidirectional=False)

        cookie = self._generate_cookie()

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
                switch=switch, match=match, queue_id=queue_id,
                priority=self.PRIORITY_REROUTE, cookie=cookie,
                installed_at=time.time(), output_port=path.egress_port,
                is_rerouted=True,
            )
            self.stats["reroutes"] += 1

            reverse_switch = "s4" if switch == "s1" else "s1"
            reverse_match = match.to_reverse_match()
            return_port = 3 if path == PathChoice.PATH_A else 4

            self.install_qos_rule(
                reverse_switch, reverse_match, queue_id,
                output_port=return_port, bidirectional=False,
            )

        return success

    def clear_reroute(self, switch: str, match: FlowMatch) -> bool:
        rule_key = self._get_rule_key(switch, match)
        if rule_key in self.installed_rules:
            rule = self.installed_rules[rule_key]
            if rule.is_rerouted:
                self.delete_rule(switch, match)
                self.install_qos_rule(switch, match, rule.queue_id)
                return True
        return False

    def cleanup_stale_rules(self, max_age_seconds: float = 300) -> int:
        now = time.time()
        stale_keys = [
            key for key, rule in self.installed_rules.items()
            if (now - rule.installed_at) > max_age_seconds
        ]
        count = 0
        for key in stale_keys:
            rule = self.installed_rules[key]
            if self.delete_rule(rule.switch, rule.match, bidirectional=False):
                count += 1
        return count

    def clear_all_rules(self, switch: Optional[str] = None) -> int:
        to_delete = [
            (rule.switch, rule.match)
            for rule in self.installed_rules.values()
            if switch is None or rule.switch == switch
        ]
        count = 0
        for sw, match in to_delete:
            if self.delete_rule(sw, match, bidirectional=False):
                count += 1
        return count

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            "active_rules": len(self.installed_rules),
            "rerouted_flows": sum(1 for r in self.installed_rules.values() if r.is_rerouted),
        }


def flow_match_from_dict(flow: Dict[str, Any]) -> FlowMatch:
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
