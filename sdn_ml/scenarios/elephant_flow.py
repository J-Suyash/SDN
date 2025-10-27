"""Elephant flow detection scenario"""

import asyncio
import time
from typing import Dict, Any
from ..models.ensemble import MLEnsemble
from ..features.extractor import FeatureExtractor


class ElephantFlowScenario:
    """Demonstrate elephant flow detection and QoS enforcement"""
    
    def __init__(self):
        self.ensemble = MLEnsemble()
        self.feature_extractor = FeatureExtractor()
        self.flows = {}
    
    async def run(self):
        """Run elephant flow scenario"""
        print("Starting Elephant Flow Detection Scenario")
        print("=" * 60)
        
        normal_flows = self._create_normal_flows()
        elephant_flow = self._create_elephant_flow()
        
        all_flows = normal_flows + [elephant_flow]
        
        print("\nMonitoring flows...")
        for i in range(30):
            print(f"\n--- Timestep {i+1} ---")
            
            for flow in all_flows:
                features = self.feature_extractor.extract_flow_features(flow)
                prob, classification = self.ensemble.xgb_model.predict(features)
                
                flow_id = flow.get('id', 'unknown')
                print(f"Flow {flow_id}: {classification} (prob: {prob:.2f})")
                
                if classification == 'elephant':
                    print(f"  -> Action: Reroute to dedicated path")
                    print(f"  -> Bandwidth reservation: 900 Mbps")
            
            await asyncio.sleep(1)
        
        print("\nScenario completed!")
    
    def _create_normal_flows(self) -> list:
        """Create normal (mice) flows"""
        flows = []
        
        for i in range(50):
            flow = {
                'id': f'h1_to_h{i+1}',
                'packets': [
                    {'size': 1500, 'timestamp': j * 0.01}
                    for j in range(100)
                ]
            }
            flows.append(flow)
        
        return flows
    
    def _create_elephant_flow(self) -> Dict[str, Any]:
        """Create elephant flow"""
        return {
            'id': 'h1_to_h15_video',
            'packets': [
                {'size': 1500, 'timestamp': j * 0.001}
                for j in range(10000)
            ]
        }
