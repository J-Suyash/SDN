"""
Mininet Topology for SDN ML Project
MVP: Simple 2-host, 1-switch topology
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import os


class MVPTopology(Topo):
    """
    MVP Topology: 2 hosts connected through 1 switch

    h1 -------- s1 -------- h2
           10 Mbps

    Used for Phase 1 to validate end-to-end pipeline.
    """

    def build(self):
        # Add switch
        s1 = self.addSwitch("s1", dpid="0000000000000001")

        # Add hosts
        h1 = self.addHost("h1", ip="10.0.0.1/24")
        h2 = self.addHost("h2", ip="10.0.0.2/24")

        # Add links with bandwidth limits
        # TCLink allows traffic control (bandwidth, delay, loss)
        self.addLink(h1, s1, cls=TCLink, bw=10)  # 10 Mbps
        self.addLink(h2, s1, cls=TCLink, bw=10)  # 10 Mbps


def run_topology():
    """Start the MVP topology with Faucet controller"""

    setLogLevel("info")

    # In hybrid mode (Faucet in Docker, Mininet on Host),
    # we connect to 127.0.0.1 (mapped to host)
    faucet_host = "127.0.0.1"
    faucet_port = 6653

    info(f"*** Connecting to Faucet at {faucet_host}:{faucet_port}\n")

    # Create topology
    topo = MVPTopology()

    # Create network with remote controller (Faucet)
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(
            name, ip=faucet_host, port=faucet_port
        ),
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
    )

    # Start network
    net.start()

    info("*** Network started\n")
    info("*** Hosts: h1 (10.0.0.1), h2 (10.0.0.2)\n")
    info("*** Testing connectivity...\n")

    # Test connectivity
    net.pingAll()

    # Start CLI for interactive testing
    CLI(net)

    # Cleanup
    net.stop()


if __name__ == "__main__":
    run_topology()
