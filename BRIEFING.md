<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# you have failed to mention how to generate the traffic, iperf ?

i want application based priority aswell, please try to remember our whole conversation and add additional points to the above briefing doc.
should clear mention

- what to do ?
- why are we doing it ?
- how to do it ?

Below is an **updated production-grade project briefing** that explicitly adds (1) **traffic generation** (iperf + realistic app traffic), (2) **application-based priority** (SNI/Host + flow features), and (3) clear **What / Why / How** for each task.

***

# Project Briefing (v2): ML‑Driven Adaptive Traffic Management in SDN (Faucet + Mininet)

## 1) Purpose and scope

### What to do

Build an SDN traffic management system that (a) **classifies flows by application/priority**, (b) **predicts congestion ahead of time**, and (c) **enforces routing/QoS policies** accordingly in a Faucet-controlled OpenFlow network.[^1][^2]

### Why we’re doing it

Static policies (fixed routing/queues) cannot adapt to real network dynamics (bursts, diurnal peaks, mixed applications, encryption) and therefore struggle to guarantee QoS for critical traffic.[^2]
Using flow statistics and lightweight metadata (like SNI), ML can drive **priority-aware** and **proactive** control decisions.[^3][^2]

### How we’ll do it

- Use **Mininet + Open vSwitch** to emulate the SDN topology and enforce link limits.[^1]
- Use **Faucet** as the SDN controller and rely on **Prometheus metrics** and/or OVS stats for telemetry collection. (Faucet exports rich counters suitable for ML features.)
- Build a Python “**Orchestrator**” service that consumes telemetry, runs ML inference, and applies policy by updating Faucet configuration or QoS settings.

***

## 2) System architecture (production-style)

### What to do

Implement these building blocks:

- **Traffic generator** (Mininet hosts)
- **Telemetry collector** (scrape Faucet metrics / OVS stats)
- **Dataset builder** (CSV/SQLite)
- **Model 1: Traffic classifier** (application/priority)
- **Model 2: Congestion predictor** (time-series)
- **Policy engine** (maps outputs → QoS/routing actions)
- **Deployment scripts** (one-command demo)


### Why we’re doing it

Separating concerns makes the system maintainable and testable (traffic generation ≠ telemetry ≠ ML training ≠ enforcement).[^2]

### How to do it

Use this repo layout:

```text
sdn_ml_project/
  mininet/
    topo.py
    traffic_profiles.py
  faucet/
    faucet.yaml
    gauge.yaml (optional)
  collector/
    scrape_prometheus.py
    build_datasets.py
  ml/
    train_classifier.py
    train_predictor.py
    infer.py
    models/
  orchestrator/
    orchestrator.py
    policy_engine.py
  data/
    flows.csv
    link_timeseries.csv
  scripts/
    run_demo.sh
```


***

## 3) Traffic generation plan (must-have for progress)

### 3.1 Iperf (baseline load + congestion)

#### What to do

Use **iperf** to generate controlled TCP/UDP flows for baseline throughput, congestion creation, and repeatable experiments.[^4]

#### Why we’re doing it

iperf gives deterministic, tunable traffic (bandwidth, duration, UDP vs TCP) which is perfect for validating prediction accuracy and policy impact.[^4]

#### How to do it (Mininet CLI examples)

Start servers on receiver hosts, then run clients on sender hosts.[^4]

- TCP baseline:

```bash
mininet> h1 iperf -s -p 5001 &
mininet> h2 iperf -c 10.0.0.1 -p 5001 -t 30 -i 1
```

- UDP “voice-like” steady stream (low bandwidth, consistent):

```bash
mininet> h3 iperf -s -u -p 5002 &
mininet> h4 iperf -c 10.0.0.3 -u -p 5002 -b 256k -t 60 -i 1
```

- Congestion (multiple parallel streams to saturate a link):

```bash
mininet> h1 iperf -s -p 5001 &
mininet> h3 iperf -s -p 5003 &
mininet> h2 iperf -c 10.0.0.1 -p 5001 -t 60 -b 10M &
mininet> h4 iperf -c 10.0.0.3 -p 5003 -t 60 -b 10M &
```

Notes:

- UDP mode uses `-u` and rate control via `-b`; TCP mode adapts automatically.[^4]
- You can run `ping` during iperf to show latency impact while links are saturated.[^5]


### 3.2 Application-like traffic (priority by “app”)

iperf alone does not create real “banking” vs “web” vs “voice” application signatures; it mainly creates transport-level load.[^2]
So add **application traffic** that produces identifiable metadata (SNI/Host) and realistic patterns.

#### What to do

Generate real application traffic from Mininet hosts:

- **Banking/payment-like**: HTTPS requests to a controlled endpoint/domain list
- **Web/office**: browsing-like bursts (HTTP GETs)
- **Voice/video-like**: UDP steady rate (or D-ITG VoIP profiles)


#### Why we’re doing it

To implement **application-based priority**, the classifier must learn from features available in encrypted environments (flow stats + TLS handshake metadata such as SNI).[^3][^2]

#### How to do it

Two options (choose both if time permits):

**Option A (simple, fast): curl/wget HTTPS requests**

- From Mininet host:

```bash
mininet> h2 curl -k https://example-bank.test/api/login
```

This creates TLS handshakes where SNI may be observable depending on your setup.[^3][^2]

**Option B (realistic app profiles): D‑ITG**
D‑ITG can generate traffic emulating application behaviors like VoIP, DNS, games, etc., using stochastic models for packet size and inter-departure time.[^6]
Use D‑ITG when you need “voice-like” flows beyond simple iperf UDP.[^6]

***

## 4) Application-based priority design (explicit)

### What to do

Define a **Priority Policy Map** and ensure the classifier predicts these classes:


| Class | Name | Examples | QoS intent |
| :-- | :-- | :-- | :-- |
| P3 | Banking/Payment | bank, UPI, wallet domains | Lowest delay, reserved bandwidth |
| P2 | Voice/Video | VoIP/WebRTC/meeting | Low jitter, low delay |
| P1 | Office/Web | email, docs, browsing | Best effort |
| P0 | Bulk/Background | updates, backups, downloads | Throttle under congestion |

### Why we’re doing it

This directly matches your objective: “prioritize payment/banking, then voice, etc.” and demonstrates business-driven QoS.[^2]

### How to do it (labeling strategy)

Use **SNI/Host-based labeling** where possible:

- Maintain a lookup list: `bank_domains.txt`, `voice_domains.txt`, etc.[^3][^2]
- Label flows by SNI domain when present, and fall back to statistical signatures when not.[^2]

Encrypted traffic classification commonly uses features like packet sizes, inter-arrival times, directionality, and can use SNI to label flows via lookup tables.[^2]
Datasets like CESNET TLS label services using the SNI domain in TLS ClientHello.[^3]

***

## 5) Telemetry + dataset building (how data gets parsed)

### What to do

Implement collectors to build two datasets:

1. `flows.csv` for **Model 1 classifier** (per flow)
2. `link_timeseries.csv` for **Model 2 predictor** (per link/port per interval)

### Why we’re doing it

ML training requires consistent, repeatable features aligned with online inference.[^2]

### How to do it

- Scrape counters at fixed interval T (10s/30s/60s).
- For time series, compute deltas between scrapes to get “bytes in interval”.
- For flows, build features per 5‑tuple where possible (or at least per host-pair/port profile for the demo).

***

## 6) ML Model specifications (what algorithms and why)

### Model 1: Traffic classifier (priority class)

#### What to do

Train a supervised classifier to output `{P0,P1,P2,P3}`.

#### Why we’re doing it

This enables **application-based priority** and policy enforcement even when payload is encrypted.[^2]

#### How to do it (features + algorithm)

Use these features (minimum viable):

- Packet count, byte count, duration
- bytes/packet, packets/sec, bytes/sec
- Direction ratio (tx/rx if available)
- TLS handshake metadata presence + SNI bucket (bank/voice/other)[^2]

Algorithm (v1):

- **RandomForestClassifier** (fast, strong baseline on tabular features)

Model 2: Congestion predictor (time series)

#### What to do

Predict next-interval utilization (regression) or “will exceed 80% in next N intervals” (classification).

#### Why we’re doing it

This supports proactive avoidance of predicted hot links and anticipates known patterns like “9am login surge.”[^2]

#### How to do it

- Input window: last k intervals of utilization + hour-of-day
- Algorithm v1: RandomForestRegressor on lag features
- Algorithm v2 (research upgrade): LSTM/GRU (better temporal modeling)

(Keep v1 for demo reliability; add v2 if time.)

***

## 7) Orchestrator + enforcement (how the system actually acts)

### What to do

Implement `orchestrator.py` that:

- Loads `classifier.pkl` and `predictor.pkl`
- Maintains live state: current link loads + predicted loads
- Applies policy:
    - P3/P2 avoid predicted congestion; allocate best queue
    - P0 throttle / reroute first during congestion


### Why we’re doing it

This is the “adaptive” part: not only measuring and predicting, but changing network behavior.[^2]

### How to do it (practical enforcement options)

- **QoS enforcement**: configure OVS QoS queues; map class → queue.
- **Path enforcement** (demo approach): predefine multiple paths and move low priority traffic when predictor says congestion risk rises.
- Keep enforcement deterministic for demo and log every decision (“why this flow got P3, why rerouted”).

***

## 8) Deliverables and demo checklist

### What to do

Ship these deliverables:

- `topo.py` + `traffic_profiles.py` (iperf + app-like scripts)
- `flows.csv`, `link_timeseries.csv` (sample data included)
- `train_classifier.py` producing `classifier.pkl`
- `train_predictor.py` producing `predictor.pkl`
- `orchestrator.py` running inference + policy decisions
- `run_demo.sh` that starts everything end-to-end


### Why we’re doing it

This guarantees you can show “progress” anytime: topology → traffic → dataset → model → controller action.

### How to do it (demo story)

1. Start Faucet + Mininet, show connectivity.[^1]
2. Run iperf baseline and congestion, show counters changing.[^4]
3. Run “banking” HTTPS test + “voice-like” UDP stream, show classifier output labels.[^2]
4. Show predictor forecasting congestion during a scheduled “9am burst” script (simulated schedule).
5. Show policy: banking stays low-latency, bulk is throttled/rerouted.

***

## 9) Explicit “9AM office login” scenario (traffic pattern)

### What to do

Create a scheduled traffic profile that spikes many short flows around a configured “peak time” (simulate 9AM).

### Why we’re doing it

To demonstrate the key research claim: congestion can be predicted from repeated daily patterns and mitigated proactively.[^2]

### How to do it

Implement `traffic_profiles.py` that:

- At `t=0..60s`: multiple hosts run bursty HTTP/HTTPS (login-like) + moderate iperf flows
- At `t=60..180s`: sustained load (office active)
- At `t=180..240s`: cooldown
Log timestamp so Model 2 learns periodicity via time features.

***

If you want, the next output can be a **ready-to-run “traffic_profiles.py”** that automatically launches:

- P3 “banking” flows (HTTPS requests or tagged host flows),
- P2 voice-like UDP iperf profiles,
- P0 bulk TCP iperf transfers,
and logs ground-truth labels into `flows.csv` for training.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: http://mininet.org/walkthrough/

[^2]: https://cacm.acm.org/research/traffic-classification-in-an-increasingly-encrypted-web/

[^3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11489426/

[^4]: https://onl.wustl.edu/Tutorial/Filters,_Queues_and_Bandwidth/Generating_Traffic_With_Iperf.html

[^5]: https://stackoverflow.com/questions/63907768/use-iperf-and-ping-at-the-same-time-mininet

[^6]: https://traffic.comics.unina.it/software/ITG/manual/

[^7]: https://stackoverflow.com/questions/62955140/creating-traffic-from-hosts-at-the-same-time-at-mininet

[^8]: https://groups.google.com/a/onosproject.org/g/onos-dev/c/B9jD_iIA310

[^9]: https://pages.cs.wisc.edu/~agember/cs640/s15/assign1/

[^10]: https://traffic.comics.unina.it/software/ITG/D-ITGpublications/54URCININA.pdf

[^11]: https://www.youtube.com/watch?v=WzN7xm8G2vA

[^12]: https://www.nm-2.com/ditgbox/

[^13]: https://www.cisco.com/c/en/us/td/docs/wireless/asr_5000/21-21/ADC-Admin/21-21-adc-admin/m_sni_detection.pdf

[^14]: https://www.youtube.com/watch?v=TNMSlljIikI

[^15]: https://dl.acm.org/doi/10.5555/1025129.1026100

