# Advanced ML-Driven Adaptive Traffic Management in SDN

This project implements a sophisticated ML-driven traffic management system for Software-Defined Networks using Mininet, OpenVSwitch, Faucet, and Prometheus.

## Architecture

The system uses an ensemble of 5 ML models:
1. **LSTM** - Time-series traffic prediction
2. **XGBoost** - Elephant flow detection
3. **CNN-LSTM** - Spatial-temporal pattern recognition
4. **Isolation Forest** - Anomaly detection
5. **DQN** - Dynamic routing optimization

## Features

- Real-time feature extraction from network metrics
- Multi-path load balancing with traffic prediction
- Elephant flow detection and QoS enforcement
- Application-aware traffic engineering
- Congestion avoidance with predictive rerouting
- DDoS detection and automated mitigation
- Adaptive QoS during flash crowd events

## Quick Start

```bash
uv sync
python scripts/setup_network.py
python scripts/run_scenario.py --scenario elephant_flow
```

## Project Structure

```
sdn_ml/
├── config/         # Configuration files
├── models/         # ML model implementations
├── features/       # Feature extraction
├── controller/     # SDN controller integration
├── scenarios/     # Network scenarios
└── utils/          # Utility functions
```

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed architecture and implementation details.
