"""Main traffic management orchestration"""

import asyncio
from typing import Dict, List, Any
import numpy as np

from ..models.ensemble import MLEnsemble
from ..features.extractor import FeatureExtractor
from ..features.prometheus_collector import PrometheusCollector
from .faucet_controller import FaucetController


class MLTrafficManager:
    """Main ML-driven traffic manager"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.ensemble = MLEnsemble()
        self.feature_extractor = FeatureExtractor()
        self.prometheus = PrometheusCollector(prometheus_url)
        self.controller = FaucetController()
        
        self.running = False
    
    async def start(self):
        """Start the traffic manager"""
        self.running = True
        asyncio.create_task(self._monitoring_loop())
    
    async def stop(self):
        """Stop the traffic manager"""
        self.running = False
    
    async def _monitoring_loop(self):
        """Main monitoring and decision loop"""
        while self.running:
            try:
                metrics = await self._collect_metrics()
                features = self._extract_features(metrics)
                recommendations = self.ensemble.get_recommendations(features)
                
                await self._execute_recommendations(recommendations)
                
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from Prometheus"""
        switch_ids = ['0x1', '0x2']
        return await self.prometheus.collect_all_metrics(switch_ids)
    
    def _extract_features(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from metrics"""
        features = {}
        
        for switch_id, switch_metrics in metrics.items():
            if switch_id == 'network_latency':
                continue
            
            port_features = self.feature_extractor.extract_port_features({
                'utilization': switch_metrics.get('port_stats', {}).get('total', 0),
                'queue_depth': switch_metrics.get('queue_depth', {}).get('total', 0),
                'drops': switch_metrics.get('drops', 0),
                'rx_packets': switch_metrics.get('rx_packets', 1),
            })
            
            features.update(port_features)
        
        network_features = self.feature_extractor.extract_network_features({
            'total_capacity': 100e9,
            'total_traffic': metrics.get('total_traffic', 0),
            'link_utilizations': [0.5, 0.6, 0.7],
            'failed_links': 0,
            'controller_cpu': 15.0,
        })
        
        features.update(network_features)
        
        return features
    
    async def _execute_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Execute routing recommendations"""
        for rec in recommendations:
            action = rec.get('action')
            
            if action == 'reroute':
                await self._handle_reroute(rec)
            elif action == 'alert':
                await self._handle_alert(rec)
            elif action == 'route':
                await self._handle_route(rec)
    
    async def _handle_reroute(self, rec: Dict[str, Any]):
        """Handle reroute recommendation"""
        print(f"Rerouting: {rec.get('details')}")
    
    async def _handle_alert(self, rec: Dict[str, Any]):
        """Handle alert recommendation"""
        print(f"ALERT: {rec.get('details')}")
    
    async def _handle_route(self, rec: Dict[str, Any]):
        """Handle route recommendation"""
        print(f"Routing: {rec.get('details')}")
