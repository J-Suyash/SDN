# ML-Driven Adaptive Traffic Management in SDN

A production-grade SDN traffic management system that uses machine learning to:
- **Classify network flows** by application/priority (Banking, Voice, Web, Bulk)
- **Predict congestion** before it happens
- **Enforce QoS policies** dynamically via Faucet SDN controller

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd SDN

# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Run MVP demo
./scripts/run_mvp.sh

# Run full demo (after training models)
./scripts/run_demo.sh
```

## Project Structure

```
.
├── docker/                 # Docker configurations
│   ├── docker-compose.yml  # Main compose file
│   ├── mininet/           # Mininet + OVS container
│   ├── faucet/            # Faucet SDN controller
│   ├── prometheus/        # Metrics collection
│   └── ml-orchestrator/   # ML inference container
├── mininet/               # Topology and traffic generation
│   ├── topo.py            # Network topology
│   └── traffic_profiles.py # Traffic patterns
├── faucet/                # Faucet configuration
│   └── faucet.yaml        # SDN rules
├── collector/             # Telemetry collection
│   ├── scrape_ovs.py      # OVS statistics
│   ├── scrape_prometheus.py # Prometheus metrics
│   └── build_datasets.py  # Dataset generation
├── ml/                    # Machine learning
│   ├── notebooks/         # Colab training notebooks
│   └── models/            # Trained models (.pkl)
├── orchestrator/          # Control plane
│   ├── orchestrator.py    # Main control loop
│   ├── policy_engine.py   # QoS enforcement
│   └── stubs/             # MVP hardcoded classifiers
├── data/                  # Datasets
│   ├── domain_lists/      # Priority domain mappings
│   ├── raw/               # Raw captures
│   └── processed/         # Training CSVs
├── scripts/               # Automation scripts
├── STATUS.md              # Project status tracker
└── BRIEFING.md            # Project requirements
```

## Priority Classes

| Class | Name | Examples | QoS Intent |
|-------|------|----------|------------|
| P3 | Banking/Payment | Bank APIs, UPI, wallets | Lowest delay, reserved bandwidth |
| P2 | Voice/Video | Zoom, Teams, VoIP | Low jitter, low delay |
| P1 | Office/Web | Email, docs, browsing | Best effort |
| P0 | Bulk/Background | Updates, backups, downloads | Throttle under congestion |

## Training Models (Google Colab)

Since this project is designed for CPU-only environments, ML training is done in Google Colab:

1. **Collect data**: Run `./scripts/collect_training_data.sh`
2. **Upload to Colab**: Upload `data/processed/flows.csv` and `link_timeseries.csv`
3. **Train models**: Run notebooks in `ml/notebooks/`:
   - `01_data_exploration.ipynb` - Explore data
   - `02_train_classifier.ipynb` - Train traffic classifier
   - `03_train_predictor.ipynb` - Train congestion predictor
   - `04_model_evaluation.ipynb` - Evaluate models
4. **Download models**: Place `classifier.pkl` and `predictor.pkl` in `ml/models/`
5. **Deploy**: Restart the orchestrator container

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Mininet   │────▶│   Faucet    │────▶│ Prometheus  │
│  (Traffic)  │     │ (Controller)│     │  (Metrics)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                    ┌─────────────────────────────────────┐
                    │          ML Orchestrator            │
                    │  ┌───────────┐  ┌───────────────┐  │
                    │  │Classifier │  │   Predictor   │  │
                    │  │  (P0-P3)  │  │ (Congestion)  │  │
                    │  └───────────┘  └───────────────┘  │
                    │         ┌───────────────┐          │
                    │         │ Policy Engine │          │
                    │         └───────────────┘          │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                           QoS/Routing Actions
```

## Development Phases

See [STATUS.md](STATUS.md) for current progress.

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project scaffolding | DONE |
| 1 | MVP - End-to-end validation | PENDING |
| 2 | Full topology + traffic | PENDING |
| 3 | Telemetry pipeline | PENDING |
| 3.5 | Scapy packet capture | PENDING |
| 4 | ML training (Colab) | PENDING |
| 4.5 | ML refinement | PENDING |
| 5 | Orchestrator + policies | PENDING |
| 6 | Demo scenarios | PENDING |

## Useful Commands

```bash
# Start services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f ml-orchestrator

# Enter Mininet container
docker-compose -f docker/docker-compose.yml exec mininet bash

# Stop services
docker-compose -f docker/docker-compose.yml down

# Rebuild containers
docker-compose -f docker/docker-compose.yml build --no-cache
```

## Endpoints

- **Prometheus**: http://localhost:9090
- **Faucet metrics**: http://localhost:9302/metrics

## Requirements

- Docker & Docker Compose
- Linux host (required for Mininet)
- 4GB+ RAM
- Google account (for Colab training)

## References

- [BRIEFING.md](BRIEFING.md) - Full project specification
- [Faucet Documentation](https://docs.faucet.nz/)
- [Mininet Walkthrough](http://mininet.org/walkthrough/)
- [scikit-learn](https://scikit-learn.org/)

## License

MIT
