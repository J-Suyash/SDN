#!/usr/bin/env python3
"""Run ML SDN scenarios"""

import asyncio
import sys
import argparse

from sdn_ml.scenarios.elephant_flow import ElephantFlowScenario
from sdn_ml.scenarios.multipath_balancing import MultiPathBalancingScenario
from sdn_ml.scenarios.ddos_detection import DDoSDetectionScenario


async def main():
    operator = argparse.ArgumentParser(description='Run ML SDN scenarios')
    operator.add_argument('--scenario', choices=['elephant', 'multipath', 'ddos', 'all'],
                        default='all', help='Scenario to run')
    
    args = operator.parse_args()
    
    scenarios_to_run = []
    
    if args.scenario == 'all':
        scenarios_to_run = [
            ('Elephant Flow Detection', ElephantFlowScenario()),
            ('Multi-Path Load Balancing', MultiPathBalancingScenario()),
            ('DDoS Detection', DDoSDetectionScenario()),
        ]
    elif args.scenario == 'elephant':
        scenarios_to_run = [('Elephant Flow Detection', ElephantFlowScenario())]
    elif args.scenario == 'multipath':
        scenarios_to_run = [('Multi-Path Load Balancing', MultiPathBalancingScenario())]
    elif args.scenario == 'ddos':
        scenarios_to_run = [('DDoS Detection', DDoSDetectionScenario())]
    
    for name, scenario in scenarios_to_run:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        await scenario.run()
        print(f"\n{'='*60}\n")
    
    print("All scenarios completed!")


if __name__ == '__main__':
    asyncio.run(main())
