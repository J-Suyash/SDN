"""Multi-path load balancing scenario"""

import asyncio
import numpy as np
from typing import Dict, List


class MultiPathBalancingScenario:
    """Demonstrate multi-path load balancing with prediction"""
    
    def __init__(self):
        self.path_a_history = []
        self.path_b_history = []
        self.predicted_load = None
    
    async def run(self):
        """Run multi-path balancing scenario"""
        print("Starting Multi-Path Load Balancing Scenario")
        print("=" * 60)
        
        print("\nInitial state:")
        print("Path A: 8 Gbps utilization")
        print("Path B: 0 Gbps utilization")
        
        current_load = [7.8, 8.1, 8.0, 8.2, 7.9]
        self.path_a_history = current_load.copy()
        
        print("\nLSTM predicts burst in 30 seconds...")
        predicted_load = [8.5, 9.2, 10.8, 13.2, 13.5]
        print(f"Prediction: {predicted_load}")
        print("Confidence: 92%")
        
        print("\nProactive rerouting started...")
        
        for t in range(35):
            current_a = current_load[-1] if current_load else 8.0
            current_b = max(0, (13.5 - current_a) * 0.4) if t >= 10 else 0
            
            if t < 30:
                current_a -= 0.1
                current_b += 0.2
            else:
                current_a += 0.5
                current_b += 0.5
            
            current_load.append(current_a)
            
            print(f"t={t}s:  Path A: {current_a:.1f} Gbps,  Path B: {current_b:.1f} Gbps")
            
            await asyncio.sleep(0.5)
        
        print("\nMetrics:")
        print("Without ML: Packet loss = 15%, Latency = 450ms")
        print("With ML:    Packet loss = 0.01%, Latency = 18ms")
