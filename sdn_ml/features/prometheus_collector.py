"""Prometheus metrics collector for SDN"""

import asyncio
from typing import Dict, List, Any
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
import aiohttp


class PrometheusCollector:
    """Collect metrics from Prometheus"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.registry = CollectorRegistry()
        
        self.metrics = {
            'port_utilization': Gauge('port_utilization_bytes', 'Port utilization', ['switch', 'port'], registry=self.registry),
            'queue_depth': Gauge('queue_depth_packets', 'Queue depth', ['switch', 'port'], registry=self.registry),
            'flow_count': Gauge('flow_count_total', 'Flow count', ['switch'], registry=self.registry),
            'packet_drops': Counter('packet_drops_total', 'Packet drops', ['switch', 'port'], registry=self.registry),
            'latency': Histogram('latency_seconds', 'Network latency', ['src', 'dst'], registry=self.registry),
        }
    
    async def query_prometheus(self, query: str) -> List[Dict[str, Any]]:
        """Query Prometheus API"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.prometheus_url}/api/v1/query"
            params = {'query': query}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {}).get('result', [])
                else:
                    return []
    
    async def get_port_stats(self, switch_id: str) -> Dict[str, Any]:
        """Get port statistics"""
        query = f'faucet_port_stats_bytes_total{{dpid="{switch_id}"}}'
        results = await self.query_prometheus(query)
        
        stats = {}
        for result in results:
            port = result['metric'].get('port', 'unknown')
            value = float(result['value'][1])
            stats[port] = value
        
        return stats
    
    async def get_queue_depth(self, switch_id: str) -> Dict[str, float]:
        """Get queue depth metrics"""
        query = f'faucet_queue_depth_packets{{dpid="{switch_id}"}}'
        results = await self.query_prometheus(query)
        
        queue_depths = {}
        for result in results:
            port = result['metric'].get('port', 'unknown')
            value = float(result['value'][1])
            queue_depths[port] = value
        
        return queue_depths
    
    async def get_flow_stats(self, switch_id: str) -> Dict[str, Any]:
        """Get flow statistics"""
        query = f'faucet_flow_table_active{{dpid="{switch_id}"}}'
        results = await self.query_prometheus(query)
        
        flow_count = 0
        for result in results:
            flow_count += int(result['value'][1])
        
        return {'flow_count': flow_count}
    
    async def get_latency_metrics(self) -> Dict[str, float]:
        """Get latency metrics"""
        query = 'faucet_latency_seconds_sum'
        results = await self.query_prometheus(query)
        
        latencies = {}
        for result in results:
            src = result['metric'].get('src', 'unknown')
            dst = result['metric'].get('dst', 'unknown')
            value = float(result['value'][1])
            latencies[f"{src}_{dst}"] = value
        
        return latencies
    
    async def collect_all_metrics(self, switch_ids: List[str]) -> Dict[str, Any]:
        """Collect all metrics from switches"""
        all_metrics = {}
        
        for switch_id in switch_ids:
            switch_metrics = {
                'port_stats': await self.get_port_stats(switch_id),
                'queue_depth': await self.get_queue_depth(switch_id),
                'flow_stats': await self.get_flow_stats(switch_id),
            }
            all_metrics[switch_id] = switch_metrics
        
        all_metrics['network_latency'] = await self.get_latency_metrics()
        
        return all_metrics
