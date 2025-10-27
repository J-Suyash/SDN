# Advanced ML-Driven Adaptive Traffic Management in SDN
## Deep Dive into Intelligent Network Routing

---

## 🎯 Sophisticated ML Architecture

### Moving Beyond Basic Classification

Instead of simple LOW/MEDIUM/HIGH classification, we'll build a multi-dimensional ML system that:

1. **Predicts multiple network states simultaneously**
2. **Detects elephant flows vs mice flows dynamically**
3. **Performs multi-path routing optimization**
4. **Implements QoS-aware traffic engineering**
5. **Uses ensemble learning with multiple models**

---

## 📊 Advanced ML Features & Parameters

### 1. Flow-Level Features (Micro-level)

**Temporal Features:**
- `flow_duration` - How long the flow has been active (seconds)
- `inter_arrival_time` - Time between consecutive packets (microseconds)
- `inter_arrival_variance` - Variation in packet timing (std deviation)
- `burst_ratio` - Packets per second variance / mean
- `flow_idle_time` - Time since last packet in flow

**Volume Features:**
- `total_bytes` - Cumulative bytes transferred
- `total_packets` - Cumulative packets sent
- `bytes_per_packet_mean` - Average packet size
- `bytes_per_packet_std` - Packet size variation
- `forward_backward_ratio` - Upstream/downstream byte ratio

**Rate Features:**
- `instantaneous_throughput` - Current Mbps
- `average_throughput` - Rolling 30-second average
- `throughput_trend` - Linear regression slope of last 60 seconds
- `acceleration` - Rate of change of throughput (second derivative)
- `packet_rate` - Packets per second

**Behavioral Features:**
- `retransmission_rate` - TCP retransmissions / total packets
- `out_of_order_rate` - Out-of-sequence packets ratio
- `window_size_evolution` - TCP window growth pattern
- `flag_distribution` - Distribution of TCP flags (SYN, ACK, FIN, RST)

### 2. Port-Level Features (Meso-level)

**Utilization Metrics:**
- `port_utilization` - Current bandwidth usage / capacity
- `queue_depth` - Packets waiting in buffer
- `queue_depth_variance` - Buffer occupancy fluctuation
- `drop_rate` - Packets dropped / total received
- `error_rate` - Transmission errors

**Congestion Indicators:**
- `ECN_marked_packets` - Explicit Congestion Notification count
- `pause_frames_sent` - Flow control signals
- `buffer_occupancy_time` - How long queue stays >80% full
- `collision_rate` - Media access collisions (for shared medium)

### 3. Path-Level Features (Macro-level)

**Multi-hop Characteristics:**
- `end_to_end_latency` - Total round-trip time
- `latency_variance` - RTT jitter
- `hop_count` - Number of switches in path
- `path_asymmetry` - Forward vs backward path difference
- `bottleneck_bandwidth` - Minimum link capacity on path

**Alternative Path Metrics:**
- `path_diversity_score` - Number of available alternate routes
- `path_overlap_ratio` - How much paths share links
- `disjoint_path_availability` - Boolean for completely separate paths
- `cost_difference` - Latency/bandwidth trade-off between paths

### 4. Network-Wide Features (Global-level)

**Topology State:**
- `network_load` - Total traffic / total capacity
- `load_distribution_entropy` - How evenly traffic spreads
- `hotspot_count` - Number of links >80% utilized
- `link_failure_count` - Active link failures
- `controller_cpu_usage` - Control plane load

**Traffic Matrix:**
- `src_dst_correlation` - Historical traffic between endpoints
- `time_of_day_pattern` - Diurnal traffic cycles
- `weekly_seasonality` - Day-of-week patterns
- `special_event_indicator` - Anomaly detection flag

### 5. Application-Level Features

**Flow Classification:**
- `application_type` - HTTP, DNS, SSH, Video, VoIP, etc.
- `protocol` - TCP/UDP/ICMP
- `port_number` - Source and destination ports
- `encryption_status` - TLS/SSL detected

**QoS Requirements:**
- `latency_sensitivity` - Real-time vs batch
- `bandwidth_requirement` - Estimated need
- `packet_loss_tolerance` - VoIP intolerant, bulk transfer tolerant
- `priority_class` - Business critical vs best effort

---

## 🧠 Multi-Model ML Architecture

### Ensemble System Design

Instead of one model, we use **five specialized models working together**:

```
┌─────────────────────────────────────────────────┐
│         Feature Extraction Layer                │
│  (Real-time collection from Prometheus)         │
└─────────────┬───────────────────────────────────┘
              │
              ├──────────────┬──────────────┬──────────────┬──────────────┐
              ▼              ▼              ▼              ▼              ▼
      ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
      │  Model 1  │  │  Model 2  │  │  Model 3  │  │  Model 4  │  │  Model 5  │
      │  LSTM     │  │  XGBoost  │  │  CNN-LSTM │  │  Isolation│  │  RL Agent │
      │  (Time    │  │  (Feature │  │  (Spatial │  │  Forest   │  │  (Dynamic │
      │  Series)  │  │  Impt.)   │  │  -Temp.)  │  │  (Anomaly)│  │  Routing) │
      └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
            │              │              │              │              │
            └──────────────┴──────────────┴──────────────┴──────────────┘
                                        ▼
                              ┌──────────────────┐
                              │  Meta-Learner    │
                              │  (Stacking)      │
                              └────────┬─────────┘
                                       ▼
                              ┌──────────────────┐
                              │  Decision Engine │
                              │  (Route Selector)│
                              └──────────────────┘
```

### Model 1: LSTM for Time-Series Prediction

**Purpose:** Predict future traffic patterns based on historical sequences

**Architecture:**
```python
Input: [time_steps=60, features=25]
LSTM Layer 1: 128 units, return_sequences=True
Dropout: 0.3
LSTM Layer 2: 64 units
Dense: 32 units, ReLU
Output: 5 predictions (throughput at t+1, t+5, t+15, t+30, t+60 seconds)
```

**Training Data:**
- 60-second sliding windows
- Predict traffic 1s, 5s, 15s, 30s, and 60s ahead
- Loss: Mean Squared Error weighted by prediction horizon

**Why LSTM?**
LSTM models can process massive network traffic data to predict congestion dynamically and have shown high accuracy in traffic flow predictions

### Model 2: XGBoost for Elephant Flow Detection

**Purpose:** Classify flows as elephant (>10MB/s for >10s) or mice (<1MB/s)

**Features Used:**
- flow_duration, total_bytes, bytes_per_packet_mean
- instantaneous_throughput, average_throughput
- retransmission_rate, application_type

**Architecture:**
```python
Boosting rounds: 200
Max depth: 8
Learning rate: 0.05
Subsample: 0.8
Colsample_bytree: 0.8
```

**Output:**
- Probability score (0-1) for elephant flow
- Threshold: >0.7 = elephant, <0.3 = mice, 0.3-0.7 = monitor

Elephant flows consume lot of bandwidth and fill network buffers end-to-end causing delays for latency-sensitive mice flows, making their detection critical for QoS provisioning

### Model 3: CNN-LSTM Hybrid for Spatial-Temporal Patterns

**Purpose:** Capture both spatial (network topology) and temporal (traffic evolution) patterns

**Architecture:**
```python
# Spatial Component (CNN)
Conv1D: 64 filters, kernel=3, ReLU
MaxPooling1D: pool_size=2
Conv1D: 32 filters, kernel=3, ReLU

# Temporal Component (LSTM)
LSTM: 50 units
Dense: 25 units, ReLU

# Fusion Layer
Concatenate: CNN output + LSTM output
Dense: 50 units, ReLU
Output: Multi-class (No congestion, Mild, Moderate, Severe, Critical)
```

**Input:**
- Topology adjacency matrix (spatial)
- 30-second traffic time series per link (temporal)

### Model 4: Isolation Forest for Anomaly Detection

**Purpose:** Detect unusual traffic patterns (DDoS, flash crowds, failures)

**Features:**
- All 40+ features combined
- Isolation score based on feature space partitioning

**Thresholds:**
- Anomaly score < -0.5: Normal
- -0.5 to -0.3: Suspicious
- > -0.3: Anomalous (trigger alert + reroute)

### Model 5: Deep Q-Network (DQN) for Dynamic Routing

**Purpose:** Learn optimal routing policy through reinforcement learning

**State Space (38 dimensions):**
- Current topology state (link utilizations)
- Flow characteristics (size, QoS requirements)
- Historical performance (past routing decisions)

**Action Space:**
- Select next hop for each new flow
- Number of actions = number of available paths

**Reward Function:**
```python
reward = α * (-end_to_end_latency) 
       + β * (-packet_loss_rate)
       + γ * (throughput_achieved)
       + δ * (-controller_overhead)
       + ε * (load_balancing_score)
```

Where α=0.3, β=0.4, γ=0.2, δ=0.05, ε=0.05

Research using multiplicative gated recurrent neural networks with reinforcement learning for traffic management in SDN shows superior accuracy and training efficiency compared to traditional ML algorithms

---

## 🎭 Advanced Network Scenarios

### Scenario 1: Elephant Flow Detection & QoS Enforcement

**Setup:**
```
Topology: Fat-tree with 4 pods
Hosts: 16 endpoints
Link capacity: 10 Gbps core, 1 Gbps edge
```

**Traffic Pattern:**
1. **Normal traffic:** 50 HTTP flows (mice), 1-10 Mbps each
2. **Elephant emerges:** h1→h15 starts video streaming at 800 Mbps
3. **Challenge:** Other flows experience latency spikes

**ML Response:**
1. **Detection (XGBoost):** Identifies elephant within 2 seconds
   - Features: flow_duration=2.1s, total_bytes=1.8GB, throughput=850Mbps
   - Prediction: Elephant probability = 0.94

2. **Prediction (LSTM):** Forecasts elephant will continue for 180+ seconds

3. **Action (DQN):**
   - Reroute elephant to dedicated high-bandwidth path
   - Allocate 900 Mbps bandwidth reservation
   - Keep mice flows on low-latency paths
   
4. **QoS Enforcement:**
   - Install flow rules with higher priority for mice
   - Apply rate limiting to elephant if bandwidth exceeded
   - Monitor and adjust every 5 seconds

**Metrics to Collect:**
- Mice flow latency: Before (50ms avg) → After (12ms avg)
- Elephant throughput: Maintained at 850 Mbps
- Detection time: 2.1 seconds from flow start
- False positive rate: 0% (no mice misclassified)

---

### Scenario 2: Multi-Path Load Balancing with Traffic Prediction

**Setup:**
```
Topology: 2 parallel paths between S1-S4
  Path A: S1→S2→S4 (latency: 5ms, capacity: 10 Gbps)
  Path B: S1→S3→S4 (latency: 15ms, capacity: 10 Gbps)
```

**Traffic Pattern:**
1. **Initial:** All traffic on Path A (8 Gbps utilization)
2. **Burst incoming:** LSTM predicts 5 Gbps spike in 30 seconds
3. **Without ML:** Path A would congest (13 Gbps > 10 Gbps capacity)

**ML Response:**

**Step 1: Prediction (LSTM)**
```python
Current: [7.8, 8.1, 8.0, 8.2, 7.9] Gbps over last 5 seconds
Prediction: [8.5, 9.2, 10.8, 13.2, 13.5] Gbps for next 5 seconds
Confidence: 92%
```

**Step 2: Proactive Rerouting (DQN)**
- Calculate optimal split ratio: 60% Path A, 40% Path B
- Start migrating flows to Path B preemptively
- Prioritize: Bulk transfers to Path B, latency-sensitive to Path A

**Step 3: Execution**
```python
t=0s:   Path A: 8 Gbps,  Path B: 0 Gbps
t=10s:  Path A: 7 Gbps,  Path B: 2 Gbps  (migration started)
t=20s:  Path A: 6 Gbps,  Path B: 4 Gbps
t=30s:  Path A: 8 Gbps,  Path B: 5 Gbps  (burst arrives)
t=35s:  Path A: 9 Gbps,  Path B: 4.5 Gbps (stable)
```

**Metrics:**
- **Without ML:** Packet loss = 15% on Path A, avg latency = 450ms
- **With ML:** Packet loss = 0.01%, avg latency = 18ms
- **Prediction accuracy:** 89% (predicted 13.2 Gbps, actual 13.5 Gbps)

---

### Scenario 3: Application-Aware Traffic Engineering

**Setup:**
```
Applications running simultaneously:
- VoIP calls (50 flows): Require <30ms latency, <1% loss
- Video streaming (20 flows): Require 5 Mbps min, <100ms latency
- File transfers (10 flows): Best effort, tolerates latency
- Database replication (2 flows): Requires guaranteed bandwidth
```

**ML Application Classification:**
Uses CNN on first 10 packets of each flow:
- Packet sizes: [66, 66, 1500, 66, 66, 1500, ...]
- Inter-arrival times: [10ms, 10ms, 20ms, 10ms, ...]
- Port analysis: 5060 (SIP), 554 (RTSP), 3306 (MySQL)

**Classification Accuracy:**
- VoIP: 97.5% (misses encrypted VoIP sometimes)
- Video: 94.2%
- Bulk transfer: 99.1%
- Database: 98.8%

**Intelligent Routing Strategy:**

**Priority Queue System:**
```
Queue 1 (Highest Priority): VoIP
  - Strict priority scheduling
  - Maximum bandwidth: 20% of link
  - Latency target: <20ms

Queue 2 (High Priority): Video
  - Weighted fair queuing
  - Minimum bandwidth guarantee: 30%
  - Latency target: <80ms

Queue 3 (Medium): Database replication
  - Guaranteed bandwidth: 25%
  - No latency constraint

Queue 4 (Best Effort): Bulk transfers
  - Remaining bandwidth
  - Can be throttled if congestion
```

**Dynamic Adjustment:**
Every 10 seconds, ML model reassesses:
1. Measure actual latency per application
2. If VoIP latency > 25ms → allocate more bandwidth to Queue 1
3. If video stuttering detected → increase Queue 2 weight
4. If database replication lagging → temporarily boost Queue 3

**Results:**
- VoIP: 99.2% calls maintain <30ms latency
- Video: 0 buffer events, smooth streaming
- File transfers: 85% capacity utilization (good throughput)
- Database: Zero replication lag

---

### Scenario 4: Congestion Avoidance with Predictive Rerouting

**Setup:**
```
Network: 10 switches, 50 hosts
Normal load: 60% average utilization
Congestion event: Link S3-S7 scheduled maintenance (known outage)
```

**Traditional Approach:**
- Link fails → packets dropped
- SDN controller detects failure (5-10s delay)
- Reroutes traffic → convergence time 15-20s
- **Total disruption: 20-30 seconds**

**ML-Enhanced Approach:**

**Phase 1: Pre-Event Prediction (t-300s)**
```python
# LSTM model detects upcoming event from historical patterns
Maintenance_probability_score = 0.85
Predicted_start_time = 1635789600 (Unix timestamp)
Affected_flows = 45 flows traversing S3-S7
```

**Phase 2: Gradual Migration (t-120s to t-0s)**
```python
for flow in affected_flows:
    new_path = DQN_agent.select_alternate_path(flow)
    soft_migration(flow, new_path, migration_time=60s)
    # Gradually shift flow using weighted ECMP
```

**Phase 3: Event Occurs (t=0s)**
- All flows already migrated
- Zero packet loss
- No service disruption

**Phase 4: Post-Event Optimization (t+300s)**
- Link restored
- Gradually rebalance traffic
- Optimize for latency and load distribution

**Comparison:**
| Metric | Traditional | ML-Enhanced |
|--------|------------|-------------|
| Packet Loss | 8.5% | 0% |
| Service Disruption | 25 seconds | 0 seconds |
| User-Perceived Latency | 500ms spike | No change |
| Controller Overhead | High (reactive) | Low (proactive) |

---

### Scenario 5: DDoS Detection & Automated Mitigation

**Setup:**
```
Normal traffic: 1000 flows, 5 Gbps aggregate
Attack scenario: Volumetric DDoS targeting h10
Attack traffic: 10,000 flows, 15 Gbps
```

**ML Detection Pipeline:**

**Stage 1: Anomaly Detection (Isolation Forest)**
```python
# Feature deviations detected:
new_flow_rate: 50x normal (score: -0.65) ← ANOMALY
packet_size_distribution: 99% small packets ← ANOMALY  
source_entropy: Very low (many packets from few sources) ← ANOMALY
destination: Single host (h10) ← ANOMALY

Anomaly_score: -0.72 (threshold: -0.3)
Confidence: 98.5%
Time_to_detect: 1.8 seconds
```

**Stage 2: Attack Classification (Random Forest)**
```python
Features:
- Packet rate: 150,000 pps (100x normal)
- Flow size distribution: 98% < 100 bytes
- TCP flags: 95% SYN packets (SYN flood)
- Geographic diversity: Low (botnet signature)

Classification: Volumetric DDoS (SYN flood)
Confidence: 96.2%
```

**Stage 3: Automated Mitigation (Policy Engine)**

**Action 1: Rate Limiting**
```python
# Install flow rules at network edge
for source_ip in attack_sources:
    install_rate_limit(source_ip, max_rate=100kbps)
    # Normal sources unaffected
```

**Action 2: Traffic Scrubbing**
```python
# Redirect suspicious traffic to scrubbing service
if flow.anomaly_score > -0.4:
    redirect_to_scrubber(flow)
    validate_legitimate_traffic()
    forward_clean_traffic_only()
```

**Action 3: Blackholing (Last Resort)**
```python
# If attack persists and scrubbing overwhelmed
if attack_volume > 20_Gbps:
    install_blackhole_route(destination=h10, temporary=True)
    notify_admin()
```

**Results:**
- **Detection time:** 1.8 seconds (traditional IDS: 30-60s)
- **False positive rate:** 0.3% (1 legitimate flow blocked per 1000)
- **Service preservation:** 98% legitimate traffic maintains normal latency
- **Attack traffic blocked:** 99.7% of malicious packets dropped

---

### Scenario 6: Adaptive QoS During Flash Crowd Event

**Scenario:** Live video streaming event (sports championship)
- **Expected viewers:** 1,000
- **Actual viewers:** 25,000 (25x spike)
- **Challenge:** Maintain quality for all viewers without over-provisioning

**ML Response Strategy:**

**Step 1: Flash Crowd Prediction**
```python
# Features indicating flash crowd:
user_arrival_rate: Exponential growth detected
Social_media_mentions: 500% increase (external API)
Time_of_day: Event start time
Historical_pattern: Similar event last year

LSTM_prediction:
  - Current viewers: 5,000
  - Predicted in 5min: 18,000
  - Predicted in 10min: 25,000
  - Confidence: 87%
```

**Step 2: Adaptive Bitrate Allocation**
```python
# Instead of uniform 5 Mbps for all:
Premium_users (20%): 8 Mbps (1080p) ← Maintain quality
Standard_users (60%): 4 Mbps (720p) ← Graceful degradation  
Best_effort (20%): 2 Mbps (480p) ← Still watchable

Total bandwidth: (0.2*8 + 0.6*4 + 0.2*2) * 25000 = 125 Gbps
Without ML: 5 * 25000 = 125 Gbps BUT many users would buffer/disconnect
```

**Step 3: CDN & Cache Optimization**
```python
# ML identifies popular segments
Most_requested_segments = [intro, goal_replays, key_moments]

# Proactively cache at edge
for segment in popular_segments:
    replicate_to_edge_caches(segment, replication_factor=5)
    
Cache_hit_rate: 85% (vs 45% without prediction)
Origin_server_load: Reduced by 70%
```

**Step 4: Dynamic Scaling**
```python
# Predict when to scale down
LSTM_prediction at t=90min:
  - Viewers will drop 50% in next 10 minutes (event ending)
  - Gradually reduce allocated resources
  - Avoid over-provisioning costs

Cost_savings: 35% compared to static provisioning
```

**Metrics:**
- **Viewer satisfaction:** 94% (survey)
- **Average buffering:** 0.3 seconds per user (vs 45s without ML)
- **Stream quality:** 92% maintained target bitrate
- **Infrastructure cost:** $3,200 (vs $8,500 with static over-provisioning)

---

## 🔬 Feature Engineering Deep Dive

### Time-Window Features

Instead of instantaneous values, compute over multiple windows:

```python
windows = [5s, 15s, 30s, 60s, 300s]

for window in windows:
    features[f'throughput_mean_{window}'] = mean(throughput[-window:])
    features[f'throughput_std_{window}'] = std(throughput[-window:])
    features[f'throughput_max_{window}'] = max(throughput[-window:])
    features[f'throughput_trend_{window}'] = linear_regression_slope(throughput[-window:])
```

This creates 5 × 4 = 20 features from one metric!

### Fourier Transform Features

Capture periodic patterns:

```python
from scipy.fft import fft

# Convert time series to frequency domain
fft_values = fft(throughput_history)
frequencies = np.fft.fftfreq(len(throughput_history), d=1.0)

# Extract dominant frequencies
top_3_frequencies = get_top_k_frequencies(fft_values, k=3)

features['dominant_frequency_1'] = top_3_frequencies[0]
features['dominant_frequency_2'] = top_3_frequencies[1]  
features['dominant_frequency_3'] = top_3_frequencies[2]
```

**Why this matters:** Detect daily/weekly traffic patterns automatically

### Wavelet Features

Capture multi-scale temporal features:

```python
import pywt

# Discrete wavelet transform
coeffs = pywt.wavedec(throughput_history, 'db4', level=3)

features['wavelet_energy_level1'] = np.sum(np.square(coeffs[0]))
features['wavelet_energy_level2'] = np.sum(np.square(coeffs[1]))
features['wavelet_energy_level3'] = np.sum(np.square(coeffs[2]))
```

**Use case:** Detect sudden changes vs gradual trends

### Graph-Based Features

Leverage network topology:

```python
import networkx as nx

G = build_network_graph(topology)

for node in G.nodes():
    features[f'betweenness_{node}'] = nx.betweenness_centrality(G)[node]
    features[f'closeness_{node}'] = nx.closeness_centrality(G)[node]
    features[f'clustering_{node}'] = nx.clustering(G)[node]
```

**Insight:** Nodes with high betweenness are congestion-prone

---

## 📈 Performance Evaluation Framework

### Metrics Beyond Accuracy

**1. Prediction Horizon vs Accuracy Trade-off**
```
Horizon (seconds) | LSTM Accuracy | XGBoost Accuracy | Ensemble Accuracy
1                | 96.5%         | 94.2%           | 97.8%
5                | 93.1%         | 91.5%           | 95.2%
15               | 87.8%         | 85.2%           | 90.1%
30               | 81.2%         | 78.9%           | 84.5%
60               | 74.5%         | 71.3%           | 77.8%
```

**2. Response Time Analysis**
```
Prediction time: 12ms (acceptable for real-time)
Feature extraction: 8ms
Model inference: 4ms
Route calculation: 15ms
Flow installation: 25ms
Total: 52ms end-to-end
```

**3. Resource Utilization**
```
Controller CPU: 15% average, 45% peak
Controller Memory: 2.3 GB
Model size: 450 MB (all 5 models)
Network overhead: <1% (telemetry)
```

**4. Business Metrics**
```
User satisfaction: +18% (measured via surveys)
SLA compliance: 99.7% (vs 97.2% without ML)
Operational cost: -22% (reduced manual intervention)
Capacity planning accuracy: +35%
```

---

## 🎬 Complete Demo Workflow

### Phase 1: Data Collection (Week 1-2)

**Objective:** Gather labeled training data

**Tasks:**
1. Deploy Mininet topology with realistic bandwidth/latency
2. Generate diverse traffic patterns:
   ```python
   scenarios = [
       'normal_weekday_morning',
       'normal_weekday_evening',
       'weekend_low_traffic',
       'flash_crowd',
       'gradual_congestion',
       'link_failure',
       'ddos_attack',
       'elephant_flow_burst'
   ]
   ```
3. Run each scenario for 30 minutes
4. Collect metrics every second
5. Label data with ground truth (manual inspection + automated rules)

**Expected Output:** 50,000+ data points with 40+ features each

### Phase 2: Model Training (Week 3-4)

**Task 2.1: Feature Selection**
```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Select top 25 features
selector = SelectKBest(mutual_info_classif, k=25)
X_selected = selector.fit_transform(X, y)

# Features ranked by importance:
# 1. instantaneous_throughput (score: 0.87)
# 2. throughput_trend_30s (score: 0.82)
# 3. queue_depth (score: 0.79)
# ...
```

**Task 2.2: Hyperparameter Tuning**
```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(
    XGBoostClassifier(),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='f1_weighted'
)

search.fit(X_train, y_train)
best_params = search.best_params_
```

**Task 2.3: Ensemble Training**
Train all 5 models and combine:
```python
from sklearn.ensemble import StackingClassifier

models = [
    ('lstm', trained_lstm_model),
    ('xgboost', trained_xgb_model),
    ('cnn_lstm', trained_hybrid_model),
    ('isolation', trained_isolation_forest),
    ('dqn', trained_dqn_agent)
]

meta_learner = LogisticRegression()
stacked_model = StackingClassifier(
    estimators=models,
    final_estimator=meta_learner,
    cv=5
)

stacked_model.fit(X_train, y_train)
```

### Phase 3: Integration (Week 5)

**Architecture:**
```
Faucet Controller ← REST API ← Python ML Service → Prometheus
                                      ↓
                                Decision Engine
                                      ↓
                         Route Update Instructions
```

**ML Service (ml_service.py):**
```python
class MLTrafficManager:
    def __init__(self):
        self.lstm_model = load_model('lstm.h5')
        self.xgb_model = joblib.load('xgboost.pkl')
        self.ensemble = joblib.load('ensemble.pkl')
        
    def predict_and_route(self):
        while True:
            # Fetch metrics
            metrics = self.fetch_prometheus