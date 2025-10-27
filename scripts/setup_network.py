#!/usr/bin/env python3
"""Setup Mininet network"""

import sys
import subprocess
import time

from sdn_ml.scenarios.topology import create_network


def setup_faucet():
    """Setup Faucet controller"""
    print("Setting up Faucet controller...")
    
    subprocess.run(['sudo', 'systemctl', 'start', 'faucet'], check=False)
    subprocess.run(['sudo', 'systemctl', 'start', 'gauge'], check=False)
    
    time.sleep(2)
    print("Faucet controller started")


def setup_prometheus():
    """Setup Prometheus"""
    print("Setting up Prometheus...")
    
    subprocess.run(['sudo', 'systemctl', 'start', 'prometheus'], check=False)
    
    time.sleep(2)
    print("Prometheus started")


def main():
    """Main setup function"""
    print("Setting up SDN ML network environment...")
    print("=" * 60)
    
    try:
        setup_faucet()
        setup_prometheus()
        
        print("\nNetwork setup complete!")
        print("You can now run scenarios with: python scripts/run_scenario.py")
        
    except Exception as e:
        print(f"Error during setup: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
