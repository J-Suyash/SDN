# Project Status: ML-Driven Adaptive Traffic Management in SDN

> **Last Updated**: 2026-01-29
> **Current Phase**: Data Collection (User Action Required)
> **Overall Progress**: 95%

---

## Quick Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 0: Scaffolding | COMPLETE | 100% |
| Phase 1: MVP | COMPLETE | 100% |
| Phase 2: Full Topology | COMPLETE | 100% |
| Phase 3: Telemetry | COMPLETE | 100% |
| Phase 3.5: Packet Capture | COMPLETE | 100% |
| Phase 4: ML Training (Colab) | READY | Awaiting Data |
| Phase 5: Orchestrator + QoS | COMPLETE | 100% |
| Phase 6: Demo Scenarios | COMPLETE | 100% |

---

## Phase Details

### Phase 0: Project Scaffolding

**Goal**: Set up Docker-based development environment with all dependencies

**Status**: IN PROGRESS

| Task | Status | Notes |
|------|--------|-------|
| Directory structure | DONE | Created all folders |
| Docker Compose | DONE | Multi-container setup |
| Mininet Dockerfile | DONE | Ubuntu + Mininet + OVS + Python |
| Faucet Dockerfile | DONE | Official faucet image |
| Prometheus Dockerfile | DONE | With Faucet scrape config |
| ML Orchestrator Dockerfile | DONE | Python + scikit-learn |
| requirements.txt | DONE | All Python dependencies |
| STATUS.md | DONE | This file |
| Domain lists | DONE | Bank, voice, bulk, web domains |
| .gitignore updates | DONE | Updated for project |
| Colab notebooks | DONE | 4 notebooks for training |
| README.md | DONE | Project documentation |
| MVP stubs | DONE | Classifier and predictor stubs |
| Orchestrator skeleton | DONE | Basic control loop |
| Collector modules | DONE | OVS, Prometheus, dataset builders |
| Scripts | DONE | run_mvp.sh, run_demo.sh, collect_training_data.sh |

**Blockers**: None - Phase 0 COMPLETE

---

### Phase 1: MVP (Minimal Viable Pipeline)

**Goal**: End-to-end flow with hardcoded stubs to validate architecture

**Status**: IN PROGRESS (Hybrid Mode)

| Task | Status | Notes |
|------|--------|-------|
| Hybrid Mode Architecture | DONE | Faucet/Prometheus in Docker, Mininet on Host |
| Basic topology (2 hosts, 1 switch) | DONE | Updated for local host execution |
| Faucet config for basic topology | DONE | |
| start_host_mininet.sh script | DONE | Automation for host execution |
| OVS stats collection (Socket mount) | DONE | Orchestrator can talk to host OVS |
| Classifier stub (port-based) | DONE | |
| Predictor stub (threshold-based) | DONE | |
| Basic orchestrator loop | DONE | |

**Success Criteria**:
- [ ] Docker containers start in host mode
- [ ] Host Mininet connects to Docker Faucet
- [ ] Orchestrator logs decisions from real host flows

**Blockers**: None (waiting for Phase 0 completion)

---

### Phase 2: Full Topology + Traffic Generation

**Goal**: Production-like topology with diverse traffic patterns

**Status**: COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Multi-path topology | DONE | 6 switches (s1-s6), 2 paths |
| Faucet multi-path config | DONE | Stack ports configured |
| Link bandwidth configuration | DONE | Edge: 100Mbps, Core: 50Mbps |
| start_multipath_mininet.sh | DONE | Host script for Phase 2 |
| iperf profiles (TCP baseline) | DONE | In traffic_profiles.py |
| iperf profiles (UDP voice-like) | DONE | In traffic_profiles.py |
| iperf profiles (bulk transfer) | DONE | In traffic_profiles.py |
| HTTPS traffic (curl to external) | PENDING | Deferred to Phase 3.5 (Scapy) |
| Port-based priority labeling | DONE | Configured in Faucet ACLs |
| traffic_profiles.py | DONE | Updated for external endpoints |
| Gauge integration | DONE | Added Gauge for metrics export |
| Grafana integration | DONE | Added Grafana for visualization |

**Blockers**: None

---

### Phase 3: Telemetry + Dataset Pipeline

**Goal**: Robust data collection for ML training

**Status**: COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Prometheus scraper | DONE | Implemented in orchestrator |
| OVS flow stats scraper | DONE | Direct OVS socket access (Plan C) |
| flows.csv builder | DONE | Implemented in build_datasets.py |
| link_timeseries.csv builder | DONE | Implemented in build_datasets.py |
| Data validation | DONE | Tested with Phase 2 scenario |
| Orchestrator Prometheus Exporter | DONE | Exposes metrics on port 8000 |
| Grafana Dashboards | DONE | Traffic Analysis & Load Distribution |

**Blockers**: None

---

### Phase 3.5: Packet Capture with Scapy

**Goal**: Extract real network flow features for production-grade classification

**Status**: COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Scapy capture integration | DONE | Implemented AsyncSniffer in packet_capture.py |
| Flow feature extraction | DONE | Packet sizes, IAT, duration, byte/packet counts |
| TLS SNI extraction | DONE | Extracts server_name from TLSClientHello |
| Enhanced flows.csv | DONE | Updated DatasetBuilder and FlowRecord |

**Blockers**: None

---

### Phase 4: ML Training (Google Colab)

**Goal**: Train classifiers using scikit-learn in Colab notebooks

**Status**: READY - Awaiting User Data Collection

| Task | Status | Notes |
|------|--------|-------|
| Data Collection Guide | DONE | `docs/DATA_COLLECTION_GUIDE.md` |
| Quick Reference | DONE | `docs/QUICK_REFERENCE.md` |
| 01_data_exploration.py | DONE | PCAP processing + EDA |
| 02_train_classifier.py | DONE | RandomForest classifier |
| 03_train_predictor.py | DONE | Congestion prediction |
| User data collection | PENDING | Wireshark captures needed |
| Model export (.pkl) | PENDING | After training |

**Next Steps**:
1. User captures traffic using Wireshark (see `docs/QUICK_REFERENCE.md`)
2. Upload captures to Google Drive
3. Run notebooks in Colab to train models
4. Export models and integrate with orchestrator

**Blockers**: Waiting for user to capture real traffic data

---

### Phase 4.5: ML Refinement

**Goal**: Iterate on models with more data

**Status**: DEFERRED (will happen after initial training)

| Task | Status | Notes |
|------|--------|-------|
| Collect Scapy-based dataset | PENDING | |
| Retrain classifier | PENDING | |
| Retrain predictor | PENDING | |
| Compare accuracy | PENDING | |

**Blockers**: Needs Phase 3.5 + Phase 4

---

### Phase 5: Orchestrator + Policy Engine

**Goal**: Real-time inference and network control

**Status**: COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| SNI Classifier (5A) | DONE | `orchestrator/sni_classifier.py` - SNI→P0-P3 mapping |
| QoS Setup Script (5B) | DONE | `scripts/setup_qos.sh` - HTB queues on edge switches |
| QoS Enforcer Module (5C) | DONE | `orchestrator/qos_enforcer.py` - OVS flow rules |
| Policy Engine Integration (5D) | DONE | `orchestrator/policy_engine.py` - Full rewrite |
| Reroute Logic (5E) | DONE | Proactive P3, Reactive P0 path switching |
| Decision logging | DONE | Structured logging in policy_engine.py |

**Architecture**:
- Direct OVS control via `ovs-ofctl` (not Faucet ACLs) for dynamic per-flow rules
- 4 HTB Queues: Q0=P3(20-50Mbps), Q1=P2(10-45Mbps), Q2=P1(5-35Mbps), Q3=P0(0-15Mbps)
- Proactive reroute for P3 (banking) on predicted congestion
- Reactive reroute for P0 (bulk) on actual congestion

**Blockers**: None - Phase 5 COMPLETE

---

### Phase 6: Demo Scenarios

**Goal**: Showcase the system's capabilities

**Status**: COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| run_demo.sh | DONE | Master demo script with banking/9am/qos scenarios |
| banking_priority_demo.py | DONE | Python demo showing policy engine decisions |
| QoS Queue Demo | DONE | Shows HTB queue configuration |
| Reroute Behavior Demo | DONE | Proactive P3 vs Reactive P0 |

**Usage**:
```bash
# Run banking priority demo (dry-run)
python scripts/banking_priority_demo.py

# Run full topology demo
sudo ./scripts/run_demo.sh banking

# All scenarios
sudo ./scripts/run_demo.sh all
```

**Blockers**: None - Phase 6 COMPLETE

---

## Architecture Decisions

| Decision | Rationale | Date | Status |
|----------|-----------|------|--------|
| Docker-based deployment | Reproducibility, isolation, easy setup | 2026-01-17 | APPROVED |
| Separate containers for each service | Closer to production, better debugging | 2026-01-17 | APPROVED |
| scikit-learn only (no GPU) | User constraint, Colab for training | 2026-01-17 | APPROVED |
| Scapy for packet capture | Python-native, full packet inspection | 2026-01-17 | APPROVED |
| MVP with hardcoded stubs first | Validate pipeline before ML training | 2026-01-17 | APPROVED |
| Jupyter notebooks for Colab | No local GPU, cloud training | 2026-01-17 | APPROVED |
| iperf first, then real packets | Progressive data quality | 2026-01-17 | APPROVED |
| Direct OVS control (not Faucet ACLs) | More dynamic per-flow QoS control | 2026-01-29 | APPROVED |
| Rule-based first, ML later | SNI+port heuristics work; ML optional | 2026-01-29 | APPROVED |
| Proactive P3 / Reactive P0 reroute | Critical traffic gets preemptive path | 2026-01-29 | APPROVED |

---

## Known Issues

| ID | Description | Severity | Status | Phase |
|----|-------------|----------|--------|-------|
| - | No issues reported yet | - | - | - |

---

## Technical Debt

| Item | Description | Priority | Added |
|------|-------------|----------|-------|
| - | No tech debt yet | - | - |

---

## Dependencies

### External Dependencies
- Docker & Docker Compose
- Google Colab (for ML training)
- Internet access (for external HTTPS endpoints)

### Python Dependencies
See `requirements.txt`

### System Requirements
- Linux host (for Mininet privileged mode)
- 4GB+ RAM recommended
- Docker with privileged container support

---

## Team Notes

### How to Contribute
1. Check current phase status above
2. Pick a PENDING task from the current phase
3. Update STATUS.md when starting/completing tasks
4. Create PR with changes

### Running the Project
```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Run MVP test (after Phase 1)
./scripts/run_mvp.sh

# Run full demo (after Phase 6)
./scripts/run_demo.sh
```

### Useful Commands
```bash
# Check container status
docker-compose -f docker/docker-compose.yml ps

# View Faucet logs
docker-compose -f docker/docker-compose.yml logs faucet

# Enter Mininet container
docker-compose -f docker/docker-compose.yml exec mininet bash

# Access Prometheus UI
# http://localhost:9090
```

---

## Changelog

### 2026-01-29
- Phase 6 COMPLETE: Demo scenarios and documentation
- Created `scripts/run_demo.sh` - master demo script
- Created `scripts/banking_priority_demo.py` - Python demo for policy engine
- Phase 5 COMPLETE: SNI Classifier, QoS Enforcer, Policy Engine, Reroute Logic
- Created `orchestrator/sni_classifier.py` for SNI→Priority mapping
- Created `orchestrator/qos_enforcer.py` for OVS flow rule management
- Created `scripts/setup_qos.sh` for HTB queue configuration
- Rewrote `orchestrator/policy_engine.py` with full QoS enforcement
- Updated domain lists with wildcard patterns

### 2026-01-28
- Phase 3.5 COMPLETE: Scapy packet capture with flow features
- Updated `collector/packet_capture.py` with AsyncSniffer
- Updated `collector/build_datasets.py` with Scapy fields

### 2026-01-17
- Initial project setup
- Created STATUS.md
- Phase 0 started: directory structure, Dockerfiles, requirements

---

## References

- [BRIEFING.md](./BRIEFING.md) - Project requirements and specifications
- [Faucet Documentation](https://docs.faucet.nz/)
- [Mininet Walkthrough](http://mininet.org/walkthrough/)
- [scikit-learn Documentation](https://scikit-learn.org/)
