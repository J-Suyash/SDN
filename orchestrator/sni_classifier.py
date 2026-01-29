"""
SNI-Based Traffic Classifier for SDN QoS Enforcement.

Maps TLS Server Name Indication (SNI) hostnames to priority classes (P0-P3)
using domain lists with wildcard support.

Priority Classes:
    P3 - Banking/Payment (highest priority, reserved bandwidth)
    P2 - Voice/Video (low jitter, low latency)
    P1 - Web/Office (best effort, default)
    P0 - Bulk/Background (lowest priority, throttled under congestion)
"""

import os
import fnmatch
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PriorityClass(Enum):
    """Traffic priority classes with associated QoS queue IDs."""

    P0_BULK = ("P0", 3, "Bulk/Background")
    P1_WEB = ("P1", 2, "Web/Office")
    P2_VOICE = ("P2", 1, "Voice/Video")
    P3_BANKING = ("P3", 0, "Banking/Payment")

    def __init__(self, label: str, queue_id: int, description: str):
        self.label = label
        self.queue_id = queue_id
        self.description = description


@dataclass
class DomainRule:
    """A single domain matching rule."""

    pattern: str
    priority: PriorityClass
    is_wildcard: bool = False
    # Pre-compiled regex for faster matching (converted from fnmatch pattern)
    _regex: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        self.is_wildcard = "*" in self.pattern or "?" in self.pattern
        if self.is_wildcard:
            # Convert fnmatch pattern to regex for efficient repeated matching
            # fnmatch.translate returns a regex pattern string
            regex_pattern = fnmatch.translate(self.pattern)
            self._regex = re.compile(regex_pattern, re.IGNORECASE)

    def matches(self, hostname: str) -> bool:
        """Check if hostname matches this rule."""
        if not hostname:
            return False

        hostname_lower = hostname.lower()
        pattern_lower = self.pattern.lower()

        if self.is_wildcard:
            # Use pre-compiled regex for wildcard patterns
            return bool(self._regex.match(hostname_lower))
        else:
            # Exact match (case-insensitive)
            return hostname_lower == pattern_lower


class SNIClassifier:
    """
    Classifies network traffic based on TLS SNI (Server Name Indication).

    Loads domain lists from files and provides fast lookup for classification.
    Supports wildcard patterns like *.example.com.

    Usage:
        classifier = SNIClassifier("/path/to/domain_lists")
        priority = classifier.classify("www.hdfcbank.com")  # Returns "P3"
    """

    # Default paths relative to project root
    DEFAULT_DOMAIN_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "domain_lists"
    )

    # Domain list file mappings
    DOMAIN_FILES = {
        PriorityClass.P3_BANKING: "bank_domains.txt",
        PriorityClass.P2_VOICE: "voice_domains.txt",
        PriorityClass.P1_WEB: "web_domains.txt",
        PriorityClass.P0_BULK: "bulk_domains.txt",
    }

    def __init__(self, domain_dir: Optional[str] = None):
        """
        Initialize the SNI classifier.

        Args:
            domain_dir: Directory containing domain list files.
                        Defaults to data/domain_lists in project root.
        """
        self.domain_dir = domain_dir or self.DEFAULT_DOMAIN_DIR

        # Store rules by priority (higher priority checked first)
        # Order: P3 -> P2 -> P0 -> P1 (P1 is default, so checked last)
        self.rules: List[DomainRule] = []

        # Cache for repeated lookups
        self._cache: Dict[str, str] = {}
        self._cache_max_size = 10000

        # Statistics
        self.stats = {
            "lookups": 0,
            "cache_hits": 0,
            "classifications": {"P0": 0, "P1": 0, "P2": 0, "P3": 0},
        }

        # Load domain lists
        self._load_domain_lists()

    def _load_domain_lists(self) -> None:
        """Load all domain list files and build matching rules."""
        # Load in priority order (P3 first, P1 last since it's default)
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
        """
        Load a single domain list file.

        Args:
            filepath: Path to the domain list file.
            priority: Priority class to assign to domains in this file.

        Returns:
            Number of rules loaded.
        """
        count = 0
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    # Strip whitespace
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Create rule
                    rule = DomainRule(pattern=line, priority=priority)
                    self.rules.append(rule)
                    count += 1

        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")

        return count

    def classify(self, sni: str) -> str:
        """
        Classify a hostname/SNI to a priority class.

        Args:
            sni: The Server Name Indication (hostname) from TLS ClientHello,
                 or any hostname to classify.

        Returns:
            Priority class label: "P0", "P1", "P2", or "P3"
        """
        self.stats["lookups"] += 1

        # Handle missing/empty SNI
        if not sni or sni.lower() in ("unknown", "none", ""):
            return "P1"  # Default to best-effort

        sni_lower = sni.lower()

        # Check cache first
        if sni_lower in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[sni_lower]

        # Search through rules (already ordered by priority)
        result = "P1"  # Default
        for rule in self.rules:
            if rule.matches(sni_lower):
                result = rule.priority.label
                break

        # Update cache (with size limit)
        if len(self._cache) < self._cache_max_size:
            self._cache[sni_lower] = result

        # Update stats
        self.stats["classifications"][result] += 1

        return result

    def classify_with_details(self, sni: str) -> Tuple[str, int, str]:
        """
        Classify SNI and return full details.

        Args:
            sni: The Server Name Indication hostname.

        Returns:
            Tuple of (priority_label, queue_id, description)
        """
        label = self.classify(sni)

        # Map label back to PriorityClass
        label_to_class = {
            "P0": PriorityClass.P0_BULK,
            "P1": PriorityClass.P1_WEB,
            "P2": PriorityClass.P2_VOICE,
            "P3": PriorityClass.P3_BANKING,
        }

        priority_class = label_to_class[label]
        return (label, priority_class.queue_id, priority_class.description)

    def get_queue_id(self, sni: str) -> int:
        """
        Get the OVS queue ID for a given SNI.

        Args:
            sni: The Server Name Indication hostname.

        Returns:
            Queue ID (0-3, where 0 is highest priority)
        """
        _, queue_id, _ = self.classify_with_details(sni)
        return queue_id

    def add_rule(self, pattern: str, priority: str) -> None:
        """
        Dynamically add a classification rule.

        Args:
            pattern: Domain pattern (can include wildcards)
            priority: Priority label ("P0", "P1", "P2", "P3")
        """
        label_to_class = {
            "P0": PriorityClass.P0_BULK,
            "P1": PriorityClass.P1_WEB,
            "P2": PriorityClass.P2_VOICE,
            "P3": PriorityClass.P3_BANKING,
        }

        if priority not in label_to_class:
            raise ValueError(f"Invalid priority: {priority}")

        rule = DomainRule(pattern=pattern, priority=label_to_class[priority])

        # Insert at appropriate position based on priority
        # Higher priority rules should come first
        priority_order = {"P3": 0, "P2": 1, "P0": 2, "P1": 3}
        insert_pos = 0
        for i, existing_rule in enumerate(self.rules):
            if priority_order[priority] <= priority_order[existing_rule.priority.label]:
                insert_pos = i
                break
            insert_pos = i + 1

        self.rules.insert(insert_pos, rule)

        # Clear cache since rules changed
        self._cache.clear()

        logger.info(f"Added rule: {pattern} -> {priority}")

    def get_stats(self) -> Dict:
        """Get classification statistics."""
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
        """Clear the classification cache."""
        self._cache.clear()

    def reload(self) -> None:
        """Reload domain lists from disk."""
        self.rules.clear()
        self._cache.clear()
        self._load_domain_lists()
        logger.info("SNI classifier reloaded")


# Module-level singleton for convenience
_default_classifier: Optional[SNIClassifier] = None


def get_classifier() -> SNIClassifier:
    """Get or create the default SNI classifier instance."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = SNIClassifier()
    return _default_classifier


def classify_sni(sni: str) -> str:
    """
    Convenience function to classify an SNI hostname.

    Args:
        sni: The Server Name Indication hostname.

    Returns:
        Priority label: "P0", "P1", "P2", or "P3"
    """
    return get_classifier().classify(sni)


if __name__ == "__main__":
    # Self-test
    logging.basicConfig(level=logging.INFO)

    print("SNI Classifier Test")
    print("=" * 60)

    classifier = SNIClassifier()

    # Test cases
    test_cases = [
        # Banking (P3)
        ("www.hdfcbank.com", "P3"),
        ("netbanking.hdfcbank.com", "P3"),
        ("paypal.com", "P3"),
        ("api.stripe.com", "P3"),
        # Voice (P2)
        ("zoom.us", "P2"),
        ("meet.google.com", "P2"),
        ("teams.microsoft.com", "P2"),
        ("live.twitch.tv", "P2"),
        # Bulk (P0)
        ("update.windowsupdate.com", "P0"),
        ("files.dropbox.com", "P0"),
        ("pypi.org", "P0"),
        ("cdn.akamai.net", "P0"),
        # Web (P1)
        ("mail.gmail.com", "P1"),
        ("docs.google.com", "P1"),
        ("stackoverflow.com", "P1"),
        # Unknown (default P1)
        ("random-unknown-site.xyz", "P1"),
        ("", "P1"),
        ("unknown", "P1"),
    ]

    print("\nTest Results:")
    print("-" * 60)

    passed = 0
    failed = 0

    for sni, expected in test_cases:
        result = classifier.classify(sni)
        status = "PASS" if result == expected else "FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"  {status}: '{sni}' -> {result} (expected {expected})")

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    # Show stats
    print("\nClassifier Stats:")
    for key, value in classifier.get_stats().items():
        print(f"  {key}: {value}")
