#!/usr/bin/env python3
"""Generate training data for ML models"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_flow_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic flow data"""
    data = []
    
    for i in range(n_samples):
        flow_type = np.random.choice(['elephant', 'mice'], p=[0.2, 0.8])
        
        if flow_type == 'elephant':
            duration = np.random.uniform(10, 300)
            bytes_per_packet = np.random.uniform(1400, 1500)
            throughput = np.random.uniform(100, 1000)
        else:
            duration = np.random.uniform(0.1, 10)
            bytes_per_packet = np.random.uniform(64, 1500)
            throughput = np.random.uniform(0.1, 10)
        
        row = {
            'flow_duration': duration,
            'total_bytes': throughput * duration * 125000,
            'bytes_per_packet_mean': bytes_per_packet,
            'instantaneous_throughput': throughput,
            'average_throughput': throughput * np.random.uniform(0.8, 1.2),
            'retransmission_rate': np.random.uniform(0, 0.05),
            'application_type': np.random.randint(0, 10),
            'label': 1 if flow_type == 'elephant' else 0,
        }
        
        data.append(row)
    
    return pd.DataFrame(data)


def generate_traffic_history(n_samples: int = 1000, time_steps: int = 60) -> np.ndarray:
    """Generate traffic history for LSTM"""
    data = []
    
    for _ in range(n_samples):
        history = []
        base_value = np.random.uniform(1, 10)
        
        for t in range(time_steps):
            trend = np.sin(t / 10) * 0.5
            noise = np.random.normal(0, 0.1)
            value = base_value + trend + noise
            history.append(value)
        
        future_values = [
            history[-1] + np.random.normal(0, 0.2),
            history[-1] + np.random.normal(0, 0.3),
            history[-1] + np.random.normal(0, 0.4),
            history[-1] + np.random.normal(0, 0.5),
            history[-1] + np.random.normal(0, 0.6),
        ]
        
        data.append({
            'history': history,
            'future': future_values,
        })
    
    return np.array(data, dtype=object)


def main():
    """Generate all training data"""
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    print("Generating training data...")
    
    print("Generating flow data...")
    flow_data = generate_flow_data(10000)
    flow_data.to_csv(output_dir / 'flow_data.csv', index=False)
    print(f"Generated {len(flow_data)} flow samples")
    
    print("Generating traffic history...")
    traffic_data = generate_traffic_history(1000)
    np.save(output_dir / 'traffic_history.npy', traffic_data)
    print(f"Generated {len(traffic_data)} traffic sequences")
    
    print("\nTraining data generation complete!")
    print(f"Data saved to: {output_dir}/")


if __name__ == '__main__':
    main()
