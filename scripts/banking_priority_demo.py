#!/usr/bin/env python3
"""
Banking Priority Demo

Demonstrates how the SDN Policy Engine prioritizes banking traffic (P3) over
bulk download traffic (P0) through QoS queue assignment and proactive rerouting.

This script can be run:
1. Standalone (dry-run mode) to see policy decisions
2. With Mininet running to apply actual OVS rules

Usage:
    # Dry-run (no OVS changes)
    python scripts/banking_priority_demo.py

    # Live mode (requires Mininet)
    python scripts/banking_priority_demo.py --live

    # With custom traffic scenarios
    python scripts/banking_priority_demo.py --scenario congestion
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.policy_engine import PolicyEngine, Action
from orchestrator.sni_classifier import SNIClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_flow_table(flows: list, decisions: list) -> None:
    """Print flows and their decisions in a table format."""
    print(f"{'Flow ID':<20} {'SNI':<30} {'Priority':<8} {'Action':<15}")
    print("-" * 73)

    for flow, decision in zip(flows, decisions):
        sni = flow.get("sni", "")[:28] or "(port-based)"
        action = decision.action.value if decision else "N/A"
        priority = decision.priority if decision else flow.get("priority", "?")
        print(f"{flow['flow_id']:<20} {sni:<30} {priority:<8} {action:<15}")


def demo_classification() -> None:
    """Demonstrate SNI-based and port-based classification."""
    print_header("1. Traffic Classification Demo")

    classifier = SNIClassifier()

    # Test SNIs
    test_cases = [
        ("netbanking.hdfcbank.com", "Banking (SNI match)"),
        ("api.razorpay.com", "Banking (SNI match)"),
        ("meet.google.com", "Voice (SNI match)"),
        ("zoom.us", "Voice (SNI match)"),
        ("files.dropbox.com", "Bulk (SNI match)"),
        ("cdn.ubuntu.com", "Bulk (SNI match)"),
        ("docs.google.com", "Web (SNI match)"),
        ("github.com", "Web (SNI match)"),
        ("unknown.example.com", "Default fallback"),
    ]

    print(f"{'SNI':<35} {'Priority':<8} {'Category'}")
    print("-" * 60)

    for sni, category in test_cases:
        priority = classifier.classify(sni)
        print(f"{sni:<35} {priority:<8} {category}")

    print(f"\nClassifier statistics: {classifier.get_stats()}")


def demo_policy_decisions(dry_run: bool = True) -> None:
    """Demonstrate policy engine decisions."""
    print_header("2. Policy Engine Decisions Demo")

    engine = PolicyEngine(dry_run=dry_run)

    # Simulate mixed traffic
    flows = [
        # Banking transaction (P3)
        {
            "flow_id": "banking-txn-1",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.3",
            "src_port": 50000,
            "dst_port": 443,
            "protocol": 6,
            "sni": "netbanking.hdfcbank.com",
        },
        # VoIP call (P2)
        {
            "flow_id": "voice-call-1",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.4",
            "src_port": 50001,
            "dst_port": 5060,
            "protocol": 17,
            "sni": "meet.google.com",
        },
        # Large file download (P0)
        {
            "flow_id": "bulk-download-1",
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.3",
            "src_port": 50002,
            "dst_port": 5000,
            "protocol": 6,
            "sni": "files.dropbox.com",
        },
        # Web browsing (P1)
        {
            "flow_id": "web-browse-1",
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.4",
            "src_port": 50003,
            "dst_port": 80,
            "protocol": 6,
            "sni": "docs.google.com",
        },
    ]

    # Scenario 1: No congestion
    print("Scenario 1: No Congestion")
    print("-" * 40)

    predictions_normal = {
        "s1:3": {
            "current_utilization": 0.30,
            "is_congested": False,
            "predicted_congestion": False,
        },
    }

    decisions = engine.apply(flows, predictions_normal)
    print_flow_table(flows, decisions)

    print("\nResult: All traffic gets appropriate priority queues")
    print("  - P3 (Banking) -> Queue 0 (highest)")
    print("  - P2 (Voice)   -> Queue 1")
    print("  - P1 (Web)     -> Queue 2")
    print("  - P0 (Bulk)    -> Queue 3 (lowest)")

    # Scenario 2: Predicted congestion
    print("\n" + "-" * 40)
    print("Scenario 2: Predicted Congestion on Path A")
    print("-" * 40)

    predictions_predicted = {
        "s1:3": {
            "current_utilization": 0.72,
            "is_congested": False,
            "predicted_congestion": True,
        },
    }

    decisions = engine.apply(flows, predictions_predicted)

    for flow, decision in zip(flows, decisions):
        if decision.action == Action.REROUTE:
            print(f"  REROUTE: {flow['flow_id']} ({decision.priority})")
            print(f"           Reason: {decision.reason}")

    print("\nResult: P3 (Banking) PROACTIVELY rerouted to Path B")
    print("        Other traffic continues on Path A")

    # Scenario 3: Actual congestion
    print("\n" + "-" * 40)
    print("Scenario 3: Actual Congestion Detected")
    print("-" * 40)

    predictions_congested = {
        "s1:3": {
            "current_utilization": 0.85,
            "is_congested": True,
            "predicted_congestion": True,
        },
    }

    decisions = engine.apply(flows, predictions_congested)

    for flow, decision in zip(flows, decisions):
        if decision.action == Action.REROUTE:
            print(f"  REROUTE: {flow['flow_id']} ({decision.priority})")
            print(f"           Reason: {decision.reason}")

    print("\nResult: Both P3 (Banking) and P0 (Bulk) rerouted")
    print("        P3: Proactive (predicted) -> Path B")
    print("        P0: Reactive (actual) -> Path B")

    # Show statistics
    print("\n" + "-" * 40)
    print("Policy Engine Statistics:")
    print("-" * 40)
    print(json.dumps(engine.get_stats(), indent=2, default=str))


def demo_qos_enforcement(dry_run: bool = True) -> None:
    """Demonstrate QoS rule enforcement."""
    print_header("3. QoS Enforcement Demo")

    if dry_run:
        print("Running in DRY-RUN mode (no OVS changes)")
        print("Use --live flag to apply actual rules\n")
    else:
        print("Running in LIVE mode (will modify OVS)\n")

    engine = PolicyEngine(dry_run=dry_run)

    # Single banking flow
    banking_flow = {
        "flow_id": "priority-banking",
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.3",
        "src_port": 45678,
        "dst_port": 443,
        "protocol": 6,
        "sni": "secure.chase.com",
    }

    # Apply policy
    predictions = {
        "s1:3": {"current_utilization": 0.50, "is_congested": False},
    }

    print("Applying QoS for banking flow:")
    print(f"  Source: {banking_flow['src_ip']}:{banking_flow['src_port']}")
    print(f"  Dest:   {banking_flow['dst_ip']}:{banking_flow['dst_port']}")
    print(f"  SNI:    {banking_flow['sni']}")

    decisions = engine.apply([banking_flow], predictions)

    if decisions:
        d = decisions[0]
        print(f"\n  Decision: {d.action.value}")
        print(f"  Priority: {d.priority}")
        print(f"  Queue:    {d.parameters.get('queue', 'N/A')}")
        print(f"  Reason:   {d.reason}")

    if not dry_run:
        print("\nOVS rule applied. Check with:")
        print("  sudo ovs-ofctl dump-flows s1 | grep set_queue")


def demo_reroute_behavior(dry_run: bool = True) -> None:
    """Demonstrate proactive vs reactive rerouting."""
    print_header("4. Reroute Behavior Demo")

    print("Topology:")
    print("  Path A: s1 -> s2 -> s3 -> s4 (primary)")
    print("  Path B: s1 -> s5 -> s6 -> s4 (alternate)")
    print()

    engine = PolicyEngine(dry_run=dry_run)

    # Banking and Bulk flows
    flows = [
        {
            "flow_id": "bank-payment",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.3",
            "src_port": 40001,
            "dst_port": 443,
            "protocol": 6,
            "sni": "api.stripe.com",
        },
        {
            "flow_id": "iso-download",
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.3",
            "src_port": 40002,
            "dst_port": 5000,
            "protocol": 6,
            "sni": "releases.ubuntu.com",
        },
    ]

    # Simulate utilization increasing over time
    utilization_levels = [
        (0.50, "Normal"),
        (0.72, "Approaching threshold"),
        (0.82, "Congested"),
    ]

    print(
        f"{'Time':<8} {'Util':<8} {'State':<25} {'Banking Action':<20} {'Bulk Action'}"
    )
    print("-" * 90)

    for util, state in utilization_levels:
        predictions = {
            "s1:3": {
                "current_utilization": util,
                "is_congested": util >= 0.80,
                "predicted_congestion": util >= 0.70,
            },
        }

        decisions = engine.apply(flows, predictions)

        banking_action = next(
            (d.action.value for d in decisions if "bank" in d.flow_id), "-"
        )
        bulk_action = next(
            (d.action.value for d in decisions if "download" in d.flow_id), "-"
        )

        time_str = f"t+{utilization_levels.index((util, state)) * 10}s"
        print(
            f"{time_str:<8} {util:.0%}    {state:<25} {banking_action:<20} {bulk_action}"
        )

    print("\nBehavior Summary:")
    print("  - At 72% (predicted): P3 (Banking) reroutes PROACTIVELY")
    print("  - At 82% (actual):    P0 (Bulk) reroutes REACTIVELY")
    print("  - P2 (Voice) never reroutes (jitter-sensitive)")
    print("  - P1 (Web) never reroutes (best-effort)")


def main():
    parser = argparse.ArgumentParser(
        description="Banking Priority Demo for SDN Policy Engine"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Apply actual OVS rules (requires root and Mininet)",
    )
    parser.add_argument(
        "--scenario",
        choices=["all", "classify", "policy", "qos", "reroute"],
        default="all",
        help="Which demo scenario to run",
    )
    args = parser.parse_args()

    dry_run = not args.live

    print_header("SDN Traffic Management - Banking Priority Demo")

    if args.live:
        logger.warning("Running in LIVE mode - OVS rules will be modified!")
        time.sleep(2)
    else:
        print("Running in DRY-RUN mode (use --live to apply actual rules)\n")

    if args.scenario in ("all", "classify"):
        demo_classification()

    if args.scenario in ("all", "policy"):
        demo_policy_decisions(dry_run)

    if args.scenario in ("all", "qos"):
        demo_qos_enforcement(dry_run)

    if args.scenario in ("all", "reroute"):
        demo_reroute_behavior(dry_run)

    print_header("Demo Complete")
    print("Next steps:")
    print("  1. Run full topology: ./scripts/run_demo.sh banking")
    print("  2. View Prometheus metrics: http://localhost:9090")
    print("  3. Train ML models: See notebooks/")


if __name__ == "__main__":
    main()
