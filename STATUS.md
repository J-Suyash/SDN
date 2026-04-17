# Project Status: ML-Driven Adaptive Traffic Management in SDN

> **Last Updated**: 2026-04-18

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Scaffolding | DONE | Docker, project structure, domain lists |
| Phase 1: MVP | DONE | Basic topology, stubs, orchestrator skeleton |
| Phase 2: Full Topology | DONE | 6-switch multi-path, traffic profiles |
| Phase 3: Telemetry | DONE | Prometheus scraper, OVS scraper, dataset builder |
| Phase 3.5: Packet Capture | DONE | Scapy AsyncSniffer, TLS SNI extraction |
| Phase 4: ML Training | PARTIAL | Model trained on synthetic data (not real captures) |
| Phase 4.5: ML Integration | DONE | ML classifier with SNI/port fallback chain |
| Phase 5: Orchestrator + QoS | DONE | Policy engine, QoS enforcer, reroute logic |
| Phase 6: Demo | PARTIAL | Dry-run demo works; live demo needs Docker + Mininet |

## Known Issues

- ML model accuracy (98%) is measured on synthetic random data — not meaningful for real traffic
- Docker containers haven't been tested end-to-end recently
- `orchestrator.py` main loop requires Docker services + Mininet running simultaneously
- No congestion predictor ML model exists — uses threshold-based stub

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Direct OVS control (not Faucet ACLs) | More dynamic per-flow QoS control |
| ML → SNI → Port fallback chain | Graceful degradation when model unavailable |
| Proactive P3 / Reactive P0 reroute | Critical traffic gets preemptive path switching |
| Scapy for packet capture | Python-native, full packet inspection with SNI |
| scikit-learn only (no GPU) | Lightweight, CPU-friendly for edge deployment |
