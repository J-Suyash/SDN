#!/usr/bin/env python3
"""
SDN ML Traffic Management - Full System Demonstration
=====================================================

This script demonstrates the end-to-end pipeline of the SDN ML Traffic Management System,
showcasing the integration of Machine Learning, Policy Enforcement, and QoS controls.

Research Context:
-----------------
Traffic classification in SDN is traditionally done using Deep Packet Inspection (DPI)
or port-based heuristics. This system implements a hybrid approach:
1. Machine Learning (RandomForest): Statistical flow feature analysis (98% accuracy)
2. SNI Classification: TLS header analysis for encrypted traffic
3. Port Heuristics: Fallback mechanism

Scenario:
---------
We simulate a diverse traffic mix entering the SDN edge:
1. Banking Transaction (High Priority - P3)
2. VoIP Call (Jitter Sensitive - P2)
3. Large File Download (Bulk - P0)
4. Web Browsing (Best Effort - P1)

The system will:
1. Extract flow features
2. Classify traffic using ML (with fallback)
3. Make policy decisions
4. Generate OpenFlow QoS rules
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.policy_engine import PolicyEngine, Action
from orchestrator.ml_classifier import MLTrafficClassifier
from orchestrator.sni_classifier import SNIClassifier
from orchestrator.qos_enforcer import QoSEnforcer, PathChoice

# Configure logging to show only relevant info
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("demo")


# ANSI Colors for formatting
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD} {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")


def print_step(step_num, title):
    print(f"\n{Colors.CYAN}{Colors.BOLD}[Step {step_num}] {title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-' * 40}{Colors.ENDC}")


def simulate_flow_capture() -> List[Dict[str, Any]]:
    """
    Simulate captured flows with features extracted by Scapy.
    These features correspond to the trained ML model inputs.
    """
    return [
        # Case 1: Banking Flow (P3)
        # Characteristics: Small packets, bursty, encrypted (HTTPS)
        {
            "flow_id": "flow_banking_001",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.3",
            "src_port": 54321,
            "dst_port": 443,
            "protocol": 6,
            "sni": "netbanking.hdfcbank.com",
            # ML Features
            "packet_count": 45,
            "byte_count": 12500,
            "duration_sec": 2.1,
            "bytes_per_packet": 277.7,
            "packets_per_sec": 21.4,
            "bytes_per_sec": 5952.3,
            "pkt_len_min": 54,
            "pkt_len_max": 1200,
            "pkt_len_mean": 277.7,
            "pkt_len_std": 150.5,
            "iat_mean": 0.045,
            "iat_std": 0.02,
        },
        # Case 2: Voice Call (P2)
        # Characteristics: Consistent small packets, UDP, low jitter
        {
            "flow_id": "flow_voice_002",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.4",
            "src_port": 40000,
            "dst_port": 5060,
            "protocol": 17,  # UDP
            "sni": "meet.google.com",
            # ML Features
            "packet_count": 1500,
            "byte_count": 240000,
            "duration_sec": 60.0,
            "bytes_per_packet": 160.0,
            "packets_per_sec": 25.0,
            "bytes_per_sec": 4000.0,
            "pkt_len_min": 160,
            "pkt_len_max": 160,
            "pkt_len_mean": 160.0,
            "pkt_len_std": 0.0,  # Constant bit rate
            "iat_mean": 0.04,  # 25ms interval
            "iat_std": 0.001,  # Low jitter
        },
        # Case 3: Bulk Download (P0)
        # Characteristics: Large packets, sustained throughput, TCP
        {
            "flow_id": "flow_bulk_003",
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.3",
            "src_port": 55555,
            "dst_port": 443,
            "protocol": 6,
            "sni": "iso.ubuntu.com",
            # ML Features
            "packet_count": 5000,
            "byte_count": 7250000,
            "duration_sec": 10.0,
            "bytes_per_packet": 1450.0,  # MTU sized
            "packets_per_sec": 500.0,
            "bytes_per_sec": 725000.0,
            "pkt_len_min": 64,
            "pkt_len_max": 1500,
            "pkt_len_mean": 1450.0,
            "pkt_len_std": 50.0,
            "iat_mean": 0.002,
            "iat_std": 0.001,
        },
        # Case 4: Ambiguous Web Traffic (P1)
        # Characteristics: Mixed packet sizes, HTTP/HTTPS
        {
            "flow_id": "flow_web_004",
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.4",
            "src_port": 56000,
            "dst_port": 80,
            "protocol": 6,
            "sni": None,  # No SNI (HTTP)
            # ML Features
            "packet_count": 120,
            "byte_count": 85000,
            "duration_sec": 5.0,
            "bytes_per_packet": 708.3,
            "packets_per_sec": 24.0,
            "bytes_per_sec": 17000.0,
            "pkt_len_min": 64,
            "pkt_len_max": 1500,
            "pkt_len_mean": 708.3,
            "pkt_len_std": 400.0,
            "iat_mean": 0.05,
            "iat_std": 0.1,
        },
    ]


def main():
    print_header("SDN ML Traffic Management System - Demo")

    # 1. Initialization
    print_step(1, "Initializing System Components")

    # Initialize Policy Engine in Dry-Run mode (no actual OVS commands)
    engine = PolicyEngine(dry_run=True, use_ml=True)

    print(f"{Colors.GREEN}✓ Policy Engine initialized{Colors.ENDC}")

    if engine.ml_classifier and engine.ml_classifier.is_model_loaded():
        print(
            f"{Colors.GREEN}✓ ML Classifier loaded (Accuracy: {engine.ml_classifier.model_metadata.get('test_metrics', {}).get('accuracy', 'N/A')}){Colors.ENDC}"
        )
        print(f"  Features: {engine.ml_classifier.feature_names}")
    else:
        print(f"{Colors.FAIL}✗ ML Classifier failed to load{Colors.ENDC}")
        return

    print(f"{Colors.GREEN}✓ SNI Classifier loaded{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ QoS Enforcer initialized (Dry-Run){Colors.ENDC}")

    # 2. Traffic Simulation
    print_step(2, "Capturing Network Flows")
    flows = simulate_flow_capture()
    print(f"Captured {len(flows)} active flows for analysis.")

    df_display = pd.DataFrame(flows)[
        ["flow_id", "src_ip", "dst_port", "sni", "packet_count"]
    ]
    print("\nCaptured Flow Summary:")
    print(df_display.to_string(index=False))

    # 3. Classification
    print_step(3, "Traffic Classification (ML + Fallback)")

    classified_flows = []
    for flow in flows:
        print(f"\nAnalyzing Flow: {Colors.BOLD}{flow['flow_id']}{Colors.ENDC}")

        # We classify manually here to show details, though PolicyEngine does this internally
        result = engine.ml_classifier.classify(flow)

        priority_color = (
            Colors.GREEN
            if result.priority == "P3"
            else Colors.BLUE
            if result.priority == "P2"
            else Colors.WARNING
            if result.priority == "P0"
            else Colors.ENDC
        )

        print(f"  Method Used: {Colors.BOLD}{result.method.upper()}{Colors.ENDC}")
        print(f"  Confidence:  {result.confidence:.4f}")
        print(f"  Classified:  {priority_color}{result.priority}{Colors.ENDC}")

        if result.probabilities:
            probs = ", ".join([f"{k}:{v:.2f}" for k, v in result.probabilities.items()])
            print(f"  Probability: {probs}")

        # Add priority to flow for next step
        flow["priority"] = result.priority
        classified_flows.append(flow)

    # 4. Policy Decision & Enforcement
    print_step(4, "Policy Enforcement & QoS Rule Generation")

    # Mock empty predictions (No Congestion)
    # The user asked to exclude congestion predictor, so we assume healthy links
    mock_predictions = {
        "s1:3": {
            "current_utilization": 0.2,
            "is_congested": False,
            "predicted_congestion": False,
        },
        "s1:4": {
            "current_utilization": 0.1,
            "is_congested": False,
            "predicted_congestion": False,
        },
    }

    decisions = engine.apply(classified_flows, mock_predictions)

    print(f"Generated {len(decisions)} policy decisions.\n")

    for decision in decisions:
        flow_info = next(f for f in flows if f["flow_id"] == decision.flow_id)

        print(
            f"{Colors.BOLD}Decision for {decision.flow_id} ({decision.priority}){Colors.ENDC}"
        )
        print(f"  Reason: {decision.reason}")
        print(f"  Action: {Colors.CYAN}{decision.action.value.upper()}{Colors.ENDC}")
        print(f"  Target: Switch {Colors.BOLD}s1{Colors.ENDC} (Edge)")

        # Simulate OVS Rule
        match = f"tcp,nw_src={decision.src_ip},nw_dst={decision.dst_ip},tp_dst={decision.dst_port}"
        if decision.protocol == 17:
            match = f"udp,nw_src={decision.src_ip},nw_dst={decision.dst_ip},tp_dst={decision.dst_port}"

        actions = f"set_queue:{decision.parameters['queue']}"
        if decision.action == Action.REROUTE:
            path = decision.parameters["path"].value
            actions += f",output:{path[1]}"  # egress port
        else:
            actions += ",normal"

        print(f"  OpenFlow Rule:")
        print(
            f"    {Colors.WARNING}ovs-ofctl add-flow s1 priority=200,{match},actions={actions}{Colors.ENDC}"
        )
        print("")

    # 5. Reroute Scenario (Optional Demonstration)
    print_step(5, "Scenario: Proactive Congestion Avoidance")
    print("Simulating PREDICTED congestion on Path A...")

    # Update predictions to show congestion
    mock_predictions["s1:3"]["predicted_congestion"] = True

    # Re-run banking flow
    banking_flow = flows[0]  # P3
    print(
        f"\nRe-evaluating Banking Flow {Colors.BOLD}{banking_flow['flow_id']}{Colors.ENDC} under congestion..."
    )

    decisions = engine.apply([banking_flow], mock_predictions)
    d = decisions[0]

    if d.action == Action.REROUTE:
        print(f"{Colors.GREEN}✓ SUCCESS: Proactive Reroute Triggered!{Colors.ENDC}")
        print(f"  Reason: {d.reason}")
        print(f"  Action: {Colors.CYAN}REROUTE{Colors.ENDC} -> Path B")
    else:
        print(f"{Colors.FAIL}✗ FAILED: Did not reroute.{Colors.ENDC}")

    print_header("Demo Complete")


if __name__ == "__main__":
    main()
