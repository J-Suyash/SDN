"""Monitoring utilities"""

import asyncio
import time
from typing import Dict, List, Callable
from collections import deque


class NetworkMonitor:
    """Monitor network metrics"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.metrics_history = deque(maxlen=1000)
        self.callbacks = []
        self.running = False
    
    def register_callback(self, callback: Callable):
        """Register callback for metric updates"""
        self.callbacks.append(callback)
    
    async def start(self):
        """Start monitoring"""
        self.running = True
        asyncio.create_task(self._monitoring_loop())
    
    async def stop(self):
        """Stop monitoring"""
        self.running = False
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            metrics = await self._collect_metrics()
            self.metrics_history.append({
                'timestamp': time.time(),
                'metrics': metrics
            })
            
            for callback in self.callbacks:
                await callback(metrics)
            
            await asyncio.sleep(self.interval)
    
    async def _collect_metrics(self) -> Dict[str, float]:
        """Collect current metrics"""
        return {
            'cpu_usage': 15.0,
            'memory_usage': 2.3,
            'network_load': 0.6,
            'flow_count': 1000,
        }
    
    def get_recent_metrics(self, seconds: int = 60) -> List[Dict]:
        """Get recent metrics"""
        cutoff = time.time() - seconds
        return [m for m in self.metrics_history if m['timestamp'] > cutoff]
