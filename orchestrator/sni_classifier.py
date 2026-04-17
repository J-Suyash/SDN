import os
import fnmatch
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from orchestrator.types import PriorityClass

logger = logging.getLogger(__name__)


@dataclass
class DomainRule:
    pattern: str
    priority: PriorityClass
    is_wildcard: bool = False
    _regex: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        self.is_wildcard = "*" in self.pattern or "?" in self.pattern
        if self.is_wildcard:
            regex_pattern = fnmatch.translate(self.pattern)
            self._regex = re.compile(regex_pattern, re.IGNORECASE)

    def matches(self, hostname: str) -> bool:
        if not hostname:
            return False
        hostname_lower = hostname.lower()
        if self.is_wildcard:
            return bool(self._regex.match(hostname_lower))
        return hostname_lower == self.pattern.lower()


class SNIClassifier:
    DEFAULT_DOMAIN_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "domain_lists"
    )

    DOMAIN_FILES = {
        PriorityClass.P3_BANKING: "bank_domains.txt",
        PriorityClass.P2_VOICE: "voice_domains.txt",
        PriorityClass.P1_WEB: "web_domains.txt",
        PriorityClass.P0_BULK: "bulk_domains.txt",
    }

    def __init__(self, domain_dir: Optional[str] = None):
        self.domain_dir = domain_dir or self.DEFAULT_DOMAIN_DIR
        self.rules: List[DomainRule] = []
        self._cache: Dict[str, str] = {}
        self._cache_max_size = 10000
        self.stats = {
            "lookups": 0,
            "cache_hits": 0,
            "classifications": {"P0": 0, "P1": 0, "P2": 0, "P3": 0},
        }
        self._load_domain_lists()

    def _load_domain_lists(self) -> None:
        priority_order = [
            PriorityClass.P3_BANKING,
            PriorityClass.P2_VOICE,
            PriorityClass.P0_BULK,
            PriorityClass.P1_WEB,
        ]
        for priority in priority_order:
            filename = self.DOMAIN_FILES[priority]
            filepath = os.path.join(self.domain_dir, filename)
            if not os.path.exists(filepath):
                logger.warning(f"Domain list not found: {filepath}")
                continue
            count = self._load_file(filepath, priority)
            logger.info(f"Loaded {count} rules from {filename} -> {priority.label}")

    def _load_file(self, filepath: str, priority: PriorityClass) -> int:
        count = 0
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    rule = DomainRule(pattern=line, priority=priority)
                    self.rules.append(rule)
                    count += 1
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
        return count

    def classify(self, sni: str) -> str:
        self.stats["lookups"] += 1

        if not sni or sni.lower() in ("unknown", "none", ""):
            return "P1"

        sni_lower = sni.lower()
        if sni_lower in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[sni_lower]

        result = "P1"
        for rule in self.rules:
            if rule.matches(sni_lower):
                result = rule.priority.label
                break

        if len(self._cache) < self._cache_max_size:
            self._cache[sni_lower] = result
        self.stats["classifications"][result] += 1
        return result

    def classify_with_details(self, sni: str) -> Tuple[str, int, str]:
        label = self.classify(sni)
        priority_class = PriorityClass.from_label(label)
        return (label, priority_class.queue_id, priority_class.description)

    def get_queue_id(self, sni: str) -> int:
        _, queue_id, _ = self.classify_with_details(sni)
        return queue_id

    def get_stats(self) -> Dict:
        cache_hit_rate = 0.0
        if self.stats["lookups"] > 0:
            cache_hit_rate = self.stats["cache_hits"] / self.stats["lookups"]
        return {
            "total_rules": len(self.rules),
            "lookups": self.stats["lookups"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": f"{cache_hit_rate:.1%}",
            "cache_size": len(self._cache),
            "classifications": self.stats["classifications"].copy(),
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    def reload(self) -> None:
        self.rules.clear()
        self._cache.clear()
        self._load_domain_lists()


_default_classifier: Optional[SNIClassifier] = None


def get_classifier() -> SNIClassifier:
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = SNIClassifier()
    return _default_classifier


def classify_sni(sni: str) -> str:
    return get_classifier().classify(sni)
