# ML-Driven Adaptive Traffic Management in SDN

A Software-Defined Networking project that uses ML-based traffic classification and policy enforcement to manage network QoS. Built on Mininet, Open vSwitch, and Faucet.

The system classifies flows into priority tiers (P0-P3) using a RandomForest model trained on flow statistics (packet sizes, inter-arrival times, byte rates). When the ML model isn't confident, it falls back to TLS SNI domain matching, then port-based heuristics. A policy engine maps classifications to OVS QoS queues and can reroute traffic across a multi-path topology when congestion is detected or predicted.

## Quick Start

```bash
# Install dependencies (needs mininet and openvswitch)
uv sync

# Run the dry-run demo (no network required)
uv run python scripts/full_system_demo.py

# Run tests
uv run pytest tests/

# Start Docker services (Faucet, Prometheus, Grafana)
docker compose -f docker/docker-compose.yml up -d

# Run full demo (requires sudo for mininet)
sudo ./scripts/run_demo.sh
```

## Project Structure

```
.
├── orchestrator/          # Control plane
│   ├── types.py           # Shared enums and dataclasses
│   ├── orchestrator.py    # Main control loop
│   ├── policy_engine.py   # Classification + QoS decisions
│   ├── ml_classifier.py   # RandomForest ML classifier
│   ├── sni_classifier.py  # TLS SNI domain-based classifier
│   ├── qos_enforcer.py    # OVS flow rule management
│   └── stubs/             # Fallback classifiers (port-based)
├── collector/             # Telemetry collection
│   ├── packet_capture.py  # Scapy-based flow capture + SNI extraction
│   ├── scrape_ovs.py      # OVS statistics scraper
│   ├── scrape_prometheus.py # Prometheus metrics scraper
│   └── build_datasets.py  # CSV dataset builder
├── topologies/            # Mininet topologies
│   ├── topo.py            # Simple 2-host topology
│   ├── topo_multipath.py  # 6-switch multi-path topology
│   └── traffic_profiles.py # iperf traffic generation
├── docker/                # Docker Compose + Dockerfiles
├── faucet/                # Faucet SDN controller config
├── data/domain_lists/     # Domain lists for SNI classification
├── notebooks/             # ML training scripts
├── scripts/               # Shell scripts (demo, QoS setup)
└── tests/                 # Unit tests
```

## Priority Classes

| Class | Name | Examples | QoS Intent |
|-------|------|----------|------------|
| P3 | Banking/Payment | Bank APIs, UPI, wallets | Lowest delay, reserved bandwidth |
| P2 | Voice/Video | Zoom, Teams, VoIP | Low jitter, low delay |
| P1 | Office/Web | Email, docs, browsing | Best effort |
| P0 | Bulk/Background | Updates, backups, downloads | Throttle under congestion |

## Classification Architecture

```
ML Model (RandomForest) → SNI Domain Lookup → Port Heuristics
```

The ML classifier is primary when a trained model is available. It falls back to SNI-based classification (matching TLS ClientHello server names against domain lists), then to port-based heuristics as a last resort.

**Note on ML training**: The current model was trained on synthetic data. For meaningful results, train on real PCAP captures using `notebooks/02_train_classifier.py`.

## Network Topology

Multi-path topology with 6 switches and 4 hosts:
- **Path A**: s1 → s2 → s3 → s4 (default)
- **Path B**: s1 → s5 → s6 → s4 (alternate)
- Edge links: 100 Mbps, Core links: 50 Mbps

P3 (banking) is proactively rerouted to Path B when congestion is *predicted*. P0 (bulk) is reactively rerouted when congestion is *detected*.

## Requirements

- Linux (required for Mininet and OVS)
- Python 3.10+
- Mininet and Open vSwitch
- Docker (for Faucet, Prometheus, Grafana)

## License

MIT
