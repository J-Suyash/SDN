"""Mininet topology configurations"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel


class FatTreeTopo(Topo):
    """Fat-tree topology for elephant flow scenarios"""
    
    def __init__(self, k=4):
        super().__init__()
        
        self.core_switches = []
        self.aggregation_switches = []
        self.edge_switches = []
        self.hosts_list = []
        
        pods = k
        
        for pod in range(pods):
            for agg_idx in range(k // 2):
                agg_sw = self.addSwitch(f's{pod}-agg{agg_idx}', dpid=f'0x{pod}{agg_idx}')
                self.aggregation_switches.append(agg_sw)
            
            for edge_idx in range(k // 2):
                edge_sw = self.addSwitch(f's{pod}-edge{edge_idx}', dpid=f'0x{pod}{edge_idx}{edge_idx}')
                self.edge_switches.append(edge_sw)
        
        for i in range((k // 2) ** 2):
            core_sw = self.addSwitch(f's-core{i}', dpid=f'0x{i+10}')
            self.core_switches.append(core_sw)
        
        for pod in range(pods):
            pod_edge_start = pod * (k // 2)
            pod_agg_start = pod * (k // 2)
            
            for idx in range(k // 2):
                edge_idx = pod_edge_start + idx
                agg_idx = pod_agg_start + idx
                
                self.addLink(self.edge_switches[edge_idx], self.aggregation_switches[agg_idx], bw=1)
                
                if idx == 0:
                    host = self.addHost(f'h{pod}-{idx}', ip=f'10.0.{pod}.{idx+1}/24')
                    self.hosts_list.append(host)
                    self.addLink(host, self.edge_switches[edge_idx], bw=1)


class MultiPathTopo(Topo):
    """Multi-path topology for load balancing scenarios"""
    
    def __init__(self):
        super().__init__()
        
        switches = []
        for i in range(4):
            sw = self.addSwitch(f's{i+1}', dpid=f'0x{i+1}')
            switches.append(sw)
        
        self.addLink(switches[0], switches[1], bw=10, delay='5ms')
        self.addLink(switches[0], switches[2], bw=10, delay='15ms')
        self.addLink(switches[1], switches[3], bw=10, delay='5ms')
        self.addLink(switches[2], switches[3], bw=10, delay='5ms')
        
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
        
        self.addLink(h1, switches[0], bw=1)
        self.addLink(h2, switches[3], bw=1)


class SimpleTopo(Topo):
    """Simple topology for testing"""
    
    def __init__(self):
        super().__init__()
        
        switch = self.addSwitch('s1', dpid='0x1')
        
        for i in range(5):
            host = self.addHost(f'h{i+1}', ip=f'10.0.0.{i+1}/24')
            self.addLink(host, switch, bw=10)


def create_network(topology='simple'):
    """Create Mininet network"""
    setLogLevel('info')
    
    if topology == 'fattree':
        topo = FatTreeTopo(k=4)
    elif topology == 'multipath':
        topo = MultiPathTopo()
    else:
        topo = SimpleTopo()
    
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        autoSetMacs=True
    )
    
    net.start()
    return net
