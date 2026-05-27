"""Tests for SNI-based traffic classifier."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.sni_classifier import SNIClassifier, DomainRule
from orchestrator.types import PriorityClass


class TestDomainRule(unittest.TestCase):
    """Test the DomainRule matching logic."""

    def test_exact_match(self):
        rule = DomainRule(pattern="example.com", priority=PriorityClass.P3_BANKING)
        self.assertTrue(rule.matches("example.com"))
        self.assertTrue(rule.matches("EXAMPLE.COM"))
        self.assertFalse(rule.matches("www.example.com"))
        self.assertFalse(rule.matches("example.org"))

    def test_wildcard_prefix(self):
        rule = DomainRule(pattern="*.bank.com", priority=PriorityClass.P3_BANKING)
        self.assertTrue(rule.matches("secure.bank.com"))
        self.assertTrue(rule.matches("api.bank.com"))
        self.assertTrue(rule.is_wildcard)
        self.assertFalse(rule.matches("bank.com"))

    def test_wildcard_subdomain(self):
        rule = DomainRule(pattern="*.zoom.us", priority=PriorityClass.P2_VOICE)
        self.assertTrue(rule.matches("meet.zoom.us"))
        self.assertTrue(rule.matches("cdn.zoom.us"))
        self.assertFalse(rule.matches("zoom.us"))
        self.assertFalse(rule.matches("zoom.com"))

    def test_empty_hostname(self):
        rule = DomainRule(pattern="test.com", priority=PriorityClass.P1_WEB)
        self.assertFalse(rule.matches(""))
        self.assertFalse(rule.matches(None))

    def test_case_insensitivity(self):
        rule = DomainRule(pattern="BankOfAmerica.com", priority=PriorityClass.P3_BANKING)
        self.assertTrue(rule.matches("bankofamerica.com"))
        self.assertTrue(rule.matches("BANKOFAMERICA.COM"))
        self.assertTrue(rule.matches("BankOfAmerica.com"))


class TestSNIClassifier(unittest.TestCase):
    """Test the SNIClassifier with domain list loading and matching."""

    def setUp(self):
        # Create a temporary domain directory with test files
        self.tmpdir = tempfile.mkdtemp()
        self._write_domain_file("bank_domains.txt", [
            "bankofamerica.com",
            "*.chase.com",
            "paypal.com",
            "*.razorpay.com",
        ])
        self._write_domain_file("voice_domains.txt", [
            "*.zoom.us",
            "*.teams.microsoft.com",
            "voip.example.com",
        ])
        self._write_domain_file("bulk_domains.txt", [
            "*.cdn.net",
            "*.download.com",
            "update.example.com",
        ])
        self._write_domain_file("web_domains.txt", [
            "*.google.com",
            "*.microsoft.com",
            "github.com",
        ])
        self.classifier = SNIClassifier(domain_dir=self.tmpdir)

    def _write_domain_file(self, filename: str, domains: list):
        path = os.path.join(self.tmpdir, filename)
        with open(path, "w") as f:
            for d in domains:
                f.write(d + "\n")

    def test_banking_sni(self):
        self.assertEqual(self.classifier.classify("bankofamerica.com"), "P3")
        self.assertEqual(self.classifier.classify("secure.chase.com"), "P3")
        self.assertEqual(self.classifier.classify("paypal.com"), "P3")
        self.assertEqual(self.classifier.classify("api.razorpay.com"), "P3")

    def test_voice_sni(self):
        self.assertEqual(self.classifier.classify("meet.zoom.us"), "P2")
        self.assertEqual(self.classifier.classify("voip.example.com"), "P2")

    def test_bulk_sni(self):
        # *.cdn.net matches test.cdn.net
        self.assertEqual(self.classifier.classify("static.test.cdn.net"), "P0")
        # *.download.com matches foo.download.com
        self.assertEqual(self.classifier.classify("files.download.com"), "P0")
        # Exact match
        self.assertEqual(self.classifier.classify("update.example.com"), "P0")

    def test_web_sni(self):
        self.assertEqual(self.classifier.classify("mail.google.com"), "P1")
        self.assertEqual(self.classifier.classify("github.com"), "P1")

    def test_unknown_sni_defaults_to_web(self):
        self.assertEqual(self.classifier.classify("completely.unknown.domain"), "P1")
        self.assertEqual(self.classifier.classify(""), "P1")
        self.assertEqual(self.classifier.classify("none"), "P1")

    def test_priority_order_preserved(self):
        """If a domain matches multiple files, the priority order is P3 > P2 > P0 > P1."""
        self.assertEqual(self.classifier.classify("bankofamerica.com"), "P3")

    def test_classify_with_details(self):
        label, queue_id, description = self.classifier.classify_with_details(
            "bankofamerica.com"
        )
        self.assertEqual(label, "P3")
        self.assertEqual(queue_id, 0)
        self.assertIn("Banking", description)

        label, queue_id, description = self.classifier.classify_with_details(
            "meet.zoom.us"
        )
        self.assertEqual(label, "P2")
        self.assertEqual(queue_id, 1)

    def test_get_queue_id(self):
        self.assertEqual(self.classifier.get_queue_id("bankofamerica.com"), 0)
        self.assertEqual(self.classifier.get_queue_id("meet.zoom.us"), 1)
        self.assertEqual(self.classifier.get_queue_id("github.com"), 2)
        # P0 bulk domain maps to queue 3
        self.assertEqual(self.classifier.get_queue_id("files.download.com"), 3)

    def test_cache_hits(self):
        self.classifier.classify("bankofamerica.com")
        self.classifier.classify("bankofamerica.com")
        self.classifier.classify("bankofamerica.com")
        stats = self.classifier.get_stats()
        self.assertGreaterEqual(stats["cache_hits"], 2)

    def test_reload(self):
        self.classifier.reload()
        stats = self.classifier.get_stats()
        self.assertEqual(stats["total_rules"], 4 + 3 + 3 + 3)

    def test_clear_cache(self):
        self.classifier.classify("bankofamerica.com")
        self.classifier.clear_cache()
        stats = self.classifier.get_stats()
        self.assertEqual(stats["cache_size"], 0)


if __name__ == "__main__":
    unittest.main()
