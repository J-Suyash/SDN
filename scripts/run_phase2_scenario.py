"""
Phase 2 Demo: Multi-Path Topology + Banking Priority Scenario
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import time
import sys
import os
from functools import partial

# Import topology and profiles
# Add parent directory to path so we can import 'topologies' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topologies.topo_multipath import MultiPathTopology
from topologies.traffic_profiles import (
    run_banking_priority_demo,
    run_9am_burst_scenario,
)


def run_phase2():
    setLogLevel("info")

    # Parse arguments manually or use argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", default="banking", choices=["banking", "9am", "mixed"]
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duration in seconds (0 = infinite/interactive)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode (short duration)"
    )
    args = parser.parse_args()

    if args.test:
        args.duration = 60

    # 1. Start Topology
    info("*** Starting Multi-Path Topology\n")
    topo = MultiPathTopology()

    # Connect to local Faucet (since we are in hybrid mode)
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip="127.0.0.1", port=6653),
        switch=partial(OVSSwitch, protocols="OpenFlow13"),
        link=TCLink,
        autoSetMacs=True,
    )

    net.start()

    # Configure switches to connect to BOTH Faucet (6653) and Gauge (6654)
    info("*** Configuring switches for Faucet + Gauge\n")
    for switch in net.switches:
        # Set controllers manually using ovs-vsctl
        switch.cmd(
            f"ovs-vsctl set-controller {switch.name} tcp:127.0.0.1:6653 tcp:127.0.0.1:6654"
        )

    # 2. Start iperf servers
    info("*** Starting iperf servers on all hosts\n")
    for host in net.hosts:
        host.cmd("iperf -s -p 5000 &")  # Bulk
        host.cmd("iperf -s -p 5001 &")  # Web
        host.cmd("iperf -s -u -p 5002 &")  # Voice
        host.cmd("iperf -s -p 5003 &")  # Banking

    # Wait for network convergence
    info("*** Waiting 15s for network convergence (Faucet discovery)...\n")
    time.sleep(15)
    net.pingAll()

    # 3. Run Scenario
    info(f'\n*** Running "{args.scenario}" Scenario ***\n')

    if args.scenario == "banking":
        # h1 (banking) and h2 (bulk) -> h3 (server)
        run_banking_priority_demo(
            net, banking_host="h1", bulk_host="h2", server_ip="10.0.0.3"
        )
    elif args.scenario == "9am":
        run_9am_burst_scenario(net, ["h1", "h2", "h3", "h4"])
    elif args.scenario == "mixed":
        info("Starting mixed traffic...\n")
        # All types
        run_banking_priority_demo(
            net, banking_host="h1", bulk_host="h2", server_ip="10.0.0.3"
        )
        run_9am_burst_scenario(net, ["h1", "h2", "h3", "h4"])

    info("\n*** Scenario running. Check Orchestrator logs! ***\n")

    if args.duration > 0:
        info(f"*** Running for {args.duration} seconds ***\n")
        time.sleep(args.duration)
    else:
        info('*** Starting CLI (type "exit" to stop) ***\n')
        CLI(net)

    net.stop()


if __name__ == "__main__":
    run_phase2()
