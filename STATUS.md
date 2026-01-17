# Project Status: ML-Driven Adaptive Traffic Management in SDN

> **Last Updated**: 2026-01-17
> **Current Phase**: Phase 1 - MVP (In Progress)
> **Overall Progress**: 20%

---

## Quick Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 0: Scaffolding | COMPLETE | 100% |
| Phase 1: MVP | IN PROGRESS | 10% |
| Phase 2: Traffic Generation | NOT STARTED | 0% |
| Phase 3: Telemetry Pipeline | NOT STARTED | 0% |
| Phase 3.5: Packet Capture (Scapy) | NOT STARTED | 0% |
| Phase 4: ML Training (Colab) | NOT STARTED | 0% |
| Phase 4.5: ML Refinement | NOT STARTED | 0% |
| Phase 5: Orchestrator | NOT STARTED | 0% |
| Phase 6: Demo Scenarios | NOT STARTED | 0% |

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

**Status**: NOT STARTED

| Task | Status | Notes |
|------|--------|-------|
| Basic topology (2 hosts, 1 switch) | PENDING | |
| Faucet config for basic topology | PENDING | |
| iperf traffic generation | PENDING | |
| OVS stats collection | PENDING | |
| Classifier stub (port-based) | PENDING | |
| Predictor stub (threshold-based) | PENDING | |
| Basic orchestrator loop | PENDING | |
| run_mvp.sh script | PENDING | |

**Success Criteria**:
- [ ] Mininet topology starts with Faucet
- [ ] iperf flow runs between hosts
- [ ] Stats are collected from OVS
- [ ] Classifier stub labels flow
- [ ] Predictor stub predicts congestion
- [ ] Orchestrator logs decisions

**Blockers**: None (waiting for Phase 0 completion)

---

### Phase 2: Full Topology + Traffic Generation

**Goal**: Production-like topology with diverse traffic patterns

**Status**: NOT STARTED

| Task | Status | Notes |
|------|--------|-------|
| Multi-path topology | PENDING | 6 switches, 4 hosts |
| Link bandwidth configuration | PENDING | 10/50/100 Mbps |
| iperf profiles (TCP baseline) | PENDING | |
| iperf profiles (UDP voice-like) | PENDING | |
| iperf profiles (bulk transfer) | PENDING | |
| HTTPS traffic (curl to external) | PENDING | Banking simulation |
| Port-based priority labeling | PENDING | P0-P3 mapping |
| traffic_profiles.py | PENDING | |

**Blockers**: None

---

### Phase 3: Telemetry + Dataset Pipeline

**Goal**: Robust data collection for ML training

**Status**: NOT STARTED

| Task | Status | Notes |
|------|--------|-------|
| Prometheus scraper | PENDING | Faucet metrics |
| OVS flow stats scraper | PENDING | ovs-ofctl |
| flows.csv builder | PENDING | Per-flow features |
| link_timeseries.csv builder | PENDING | Per-link utilization |
| Data validation | PENDING | |

**Blockers**: None

---

### Phase 3.5: Packet Capture with Scapy

**Goal**: Extract real network flow features for production-grade classification

**Status**: NOT STARTED

| Task | Status | Notes |
|------|--------|-------|
| Scapy capture integration | PENDING | |
| Flow feature extraction | PENDING | Packet sizes, IAT, etc. |
| TLS SNI extraction | PENDING | For encrypted traffic |
| Enhanced flows.csv | PENDING | Real features |

**Blockers**: None

---

### Phase 4: ML Training (Google Colab)

**Goal**: Train classifiers using scikit-learn in Colab notebooks

**Status**: NOT STARTED

| Task | Status | Notes |
|------|--------|-------|
| 01_data_exploration.ipynb | PENDING | EDA |
| 02_train_classifier.ipynb | PENDING | RandomForest classifier |
| 03_train_predictor.ipynb | PENDING | Congestion prediction |
| 04_model_evaluation.ipynb | PENDING | Metrics, confusion matrix |
| Model export (.pkl) | PENDING | |

**Blockers**: Needs data from Phase 3

---

### Phase 4.5: ML Refinement

**Goal**: Retrain models with real packet features from Scapy

**Status**: NOT STARTED

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

**Status**: NOT STARTED

| Task | Status | Notes |
|------|--------|-------|
| Replace stubs with real models | PENDING | |
| QoS queue configuration | PENDING | OVS queues |
| Path selection logic | PENDING | Rerouting |
| Decision logging | PENDING | |
| Policy engine rules | PENDING | P0-P3 handling |

**Blockers**: Needs Phase 4

---

### Phase 6: Demo Scenarios

**Goal**: Showcase the system's capabilities

**Status**: NOT STARTED

| Task | Status | Notes |
|------|--------|-------|
| 9AM office login burst | PENDING | |
| Banking priority demo | PENDING | |
| run_demo.sh | PENDING | One-command demo |
| Demo documentation | PENDING | |

**Blockers**: Needs Phase 5

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
