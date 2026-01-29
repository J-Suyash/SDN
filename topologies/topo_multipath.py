"""
Multi-Path Topology for SDN ML Project
Phase 2: 6-switch topology with redundant paths for congestion avoidance.
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import os
from functools import partial


class MultiPathTopology(Topo):
    """
    6-Switch Multi-Path Topology
    
          s2 -------- s3
         /              \\
    h1 -- s1              s4 -- h4
         \\              /
          s5 -------- s6
    
    h2 -- s1
    h3 -- s4
    
    DPIDs:
    s1: 0x1 (Edge)
    s2: 0x2 (Core A)
    s3: 0x3 (Core A)
    s4: 0x4 (Edge)
    s5: 0x5 (Core B)
    s6: 0x6 (Core B)
    """

    def build(self):
        # Create Switches
        s1 = self.addSwitch("s1", dpid="0000000000000001")
        s2 = self.addSwitch("s2", dpid="0000000000000002")
        s3 = self.addSwitch("s3", dpid="0000000000000003")
        s4 = self.addSwitch("s4", dpid="0000000000000004")
        s5 = self.addSwitch("s5", dpid="0000000000000005")
        s6 = self.addSwitch("s6", dpid="0000000000000006")

        # Create Hosts
        h1 = self.addHost("h1", ip="10.0.0.1/24", mac="00:00:00:00:00:01")
        h2 = self.addHost("h2", ip="10.0.0.2/24", mac="00:00:00:00:00:02")
        h3 = self.addHost("h3", ip="10.0.0.3/24", mac="00:00:00:00:00:03")
        h4 = self.addHost("h4", ip="10.0.0.4/24", mac="00:00:00:00:00:04")

        # Add Links (Edge to Hosts)
        # 100Mbps links for hosts
        self.addLink(h1, s1, cls=TCLink, bw=100)
        self.addLink(h2, s1, cls=TCLink, bw=100)
        self.addLink(h3, s4, cls=TCLink, bw=100)
        self.addLink(h4, s4, cls=TCLink, bw=100)

        # Add Links (Core Network)
        # Path A (s1-s2-s3-s4): 50Mbps capacity
        self.addLink(s1, s2, cls=TCLink, bw=50)
        self.addLink(s2, s3, cls=TCLink, bw=50)
        self.addLink(s3, s4, cls=TCLink, bw=50)

        # Path B (s1-s5-s6-s4): 50Mbps capacity (Alternate path)
        self.addLink(s1, s5, cls=TCLink, bw=50)
        self.addLink(s5, s6, cls=TCLink, bw=50)
        self.addLink(s6, s4, cls=TCLink, bw=50)


def run_topology():
    """Start the Multi-Path topology with Faucet controller"""

    setLogLevel("info")

    # Connect to Faucet on localhost (Hybrid setup)
    faucet_host = "127.0.0.1"
    faucet_port = 6653

    info(f"*** Connecting to Faucet at {faucet_host}:{faucet_port}\n")

    topo = MultiPathTopology()

    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(
            name, ip=faucet_host, port=faucet_port
        ),
        switch=partial(OVSSwitch, protocols="OpenFlow13"),
        link=TCLink,
        autoSetMacs=True,
    )

    net.start()

    info("*** Network started\n")
    info("*** Configuration:\n")
    info("    Hosts: h1, h2 (on s1) -> h3, h4 (on s4)\n")
    info("    Path A: s1 -> s2 -> s3 -> s4\n")
    info("    Path B: s1 -> s5 -> s6 -> s4\n")

    # Initial connectivity test
    info("*** Testing connectivity (may fail until Faucet learns topology)...\n")
    net.pingAll()

    CLI(net)
    net.stop()


if __name__ == "__main__":
    run_topology()
