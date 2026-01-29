# %% [markdown]
# # SDN Traffic Classification - Data Exploration
#
# This notebook processes Wireshark PCAP captures and explores the extracted features
# for training our traffic priority classifier (P0-P3).
#
# ## Setup
#
# 1. Upload your captures folder to Google Drive
# 2. Mount Drive and update the `CAPTURES_PATH` below

# %%
# Install dependencies (run once)
# !pip install scapy pandas numpy matplotlib seaborn scikit-learn pyshark

# %%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# For PCAP processing
try:
    from scapy.all import rdpcap, IP, TCP, UDP, Raw
    from scapy.layers.tls.all import TLS, TLSClientHello

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Scapy not available, will use CSV exports")

# %%
# Mount Google Drive (for Colab)
try:
    from google.colab import drive

    drive.mount("/content/drive")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("Not in Colab, using local paths")

# %%
# Configuration - UPDATE THESE PATHS
if IN_COLAB:
    CAPTURES_PATH = "/content/drive/MyDrive/SDN_Project/captures"
    OUTPUT_PATH = "/content/drive/MyDrive/SDN_Project/processed"
else:
    CAPTURES_PATH = "./data/raw/captures"
    OUTPUT_PATH = "./data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Priority class mapping from filename prefix
PRIORITY_MAP = {
    "P0": "Bulk/Background",
    "P1": "Web/Office",
    "P2": "Voice/Video",
    "P3": "Banking/Payment",
}

# %% [markdown]
# ## 1. Load and Process Captures


# %%
def extract_sni_from_packet(packet):
    """Extract SNI from TLS ClientHello if present."""
    try:
        if packet.haslayer(TLSClientHello):
            tls = packet[TLSClientHello]
            if hasattr(tls, "ext"):
                for ext in tls.ext:
                    if hasattr(ext, "servernames"):
                        for sn in ext.servernames:
                            if hasattr(sn, "servername"):
                                return sn.servername.decode("utf-8", errors="ignore")
    except:
        pass
    return ""


def process_pcap_to_flows(pcap_path, label):
    """
    Process a PCAP file and extract flow-level features.

    Args:
        pcap_path: Path to PCAP file
        label: Priority label (P0, P1, P2, P3)

    Returns:
        List of flow dictionaries
    """
    if not SCAPY_AVAILABLE:
        print(f"Scapy not available, skipping {pcap_path}")
        return []

    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"Error reading {pcap_path}: {e}")
        return []

    # Aggregate packets into flows (5-tuple)
    flows = defaultdict(
        lambda: {"packets": [], "timestamps": [], "sizes": [], "sni": ""}
    )

    for pkt in packets:
        if not pkt.haslayer(IP):
            continue

        ip = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        proto = ip.proto

        src_port = dst_port = 0
        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport

        # Create flow key (5-tuple, bidirectional)
        flow_key = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)])) + (proto,)

        flow = flows[flow_key]
        flow["packets"].append(pkt)
        flow["timestamps"].append(float(pkt.time))
        flow["sizes"].append(len(pkt))
        flow["src_ip"] = src_ip
        flow["dst_ip"] = dst_ip
        flow["src_port"] = src_port
        flow["dst_port"] = dst_port
        flow["protocol"] = proto

        # Extract SNI
        if not flow["sni"]:
            sni = extract_sni_from_packet(pkt)
            if sni:
                flow["sni"] = sni

    # Convert flows to feature dictionaries
    flow_records = []
    for flow_key, flow_data in flows.items():
        if len(flow_data["packets"]) < 3:  # Skip very small flows
            continue

        timestamps = flow_data["timestamps"]
        sizes = flow_data["sizes"]

        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.001
        duration = max(duration, 0.001)  # Avoid division by zero

        # Calculate inter-arrival times
        iats = np.diff(sorted(timestamps)) if len(timestamps) > 1 else [0]

        record = {
            "flow_id": hash(flow_key) % (10**8),
            "src_ip": flow_data.get("src_ip", ""),
            "dst_ip": flow_data.get("dst_ip", ""),
            "src_port": flow_data.get("src_port", 0),
            "dst_port": flow_data.get("dst_port", 0),
            "protocol": flow_data.get("protocol", 6),
            "packet_count": len(sizes),
            "byte_count": sum(sizes),
            "duration_sec": duration,
            "bytes_per_packet": np.mean(sizes),
            "packets_per_sec": len(sizes) / duration,
            "bytes_per_sec": sum(sizes) / duration,
            "pkt_len_min": min(sizes),
            "pkt_len_max": max(sizes),
            "pkt_len_mean": np.mean(sizes),
            "pkt_len_std": np.std(sizes),
            "iat_mean": np.mean(iats) if len(iats) > 0 else 0,
            "iat_std": np.std(iats) if len(iats) > 0 else 0,
            "sni_domain": flow_data.get("sni", ""),
            "label": label,
            "source_file": os.path.basename(pcap_path),
        }
        flow_records.append(record)

    return flow_records


def process_csv_export(csv_path, label):
    """
    Process a Wireshark CSV export and extract flow-level features.

    Args:
        csv_path: Path to CSV file exported from Wireshark
        label: Priority label (P0, P1, P2, P3)

    Returns:
        List of flow dictionaries
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return []

    # Normalize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace(".", "_")

    # Required columns mapping (handle different Wireshark export formats)
    col_mapping = {
        "time": ["time", "frame_time_relative", "frame_time_epoch"],
        "source": ["source", "ip_src", "src"],
        "destination": ["destination", "ip_dst", "dst"],
        "protocol": ["protocol", "ip_proto"],
        "length": ["length", "frame_len", "len"],
    }

    # Find actual column names
    def find_col(options):
        for opt in options:
            if opt in df.columns:
                return opt
        return None

    time_col = find_col(col_mapping["time"])
    src_col = find_col(col_mapping["source"])
    dst_col = find_col(col_mapping["destination"])
    proto_col = find_col(col_mapping["protocol"])
    len_col = find_col(col_mapping["length"])

    if not all([src_col, dst_col, len_col]):
        print(f"Missing required columns in {csv_path}")
        print(f"Available columns: {list(df.columns)}")
        return []

    # Group by source-destination pairs (simplified flow aggregation)
    flows = defaultdict(lambda: {"timestamps": [], "sizes": []})

    for _, row in df.iterrows():
        src = str(row.get(src_col, ""))
        dst = str(row.get(dst_col, ""))

        # Skip non-IP traffic
        if not src or not dst or src == "nan" or dst == "nan":
            continue

        flow_key = tuple(sorted([src, dst]))
        flows[flow_key]["src"] = src
        flows[flow_key]["dst"] = dst

        if time_col:
            try:
                flows[flow_key]["timestamps"].append(float(row[time_col]))
            except:
                flows[flow_key]["timestamps"].append(len(flows[flow_key]["timestamps"]))

        if len_col:
            try:
                flows[flow_key]["sizes"].append(int(row[len_col]))
            except:
                flows[flow_key]["sizes"].append(64)

    # Convert to records
    flow_records = []
    for flow_key, flow_data in flows.items():
        sizes = flow_data["sizes"]
        timestamps = flow_data["timestamps"]

        if len(sizes) < 3:
            continue

        duration = max(timestamps) - min(timestamps) if timestamps else 1
        duration = max(duration, 0.001)

        iats = np.diff(sorted(timestamps)) if len(timestamps) > 1 else [0]

        record = {
            "flow_id": hash(flow_key) % (10**8),
            "src_ip": flow_data.get("src", ""),
            "dst_ip": flow_data.get("dst", ""),
            "src_port": 0,  # Not available in basic CSV
            "dst_port": 0,
            "protocol": 6,
            "packet_count": len(sizes),
            "byte_count": sum(sizes),
            "duration_sec": duration,
            "bytes_per_packet": np.mean(sizes),
            "packets_per_sec": len(sizes) / duration,
            "bytes_per_sec": sum(sizes) / duration,
            "pkt_len_min": min(sizes),
            "pkt_len_max": max(sizes),
            "pkt_len_mean": np.mean(sizes),
            "pkt_len_std": np.std(sizes),
            "iat_mean": np.mean(iats),
            "iat_std": np.std(iats),
            "sni_domain": "",
            "label": label,
            "source_file": os.path.basename(csv_path),
        }
        flow_records.append(record)

    return flow_records


# %%
def load_all_captures(captures_path):
    """
    Load all captures from the directory structure.

    Expected structure:
    captures/
      P3_banking/
        P3_hdfc_*.pcapng
      P2_voice_video/
        P2_zoom_*.pcapng
      ...
    """
    all_flows = []

    # Find all PCAP and CSV files
    pcap_files = glob.glob(os.path.join(captures_path, "**/*.pcap*"), recursive=True)
    csv_files = glob.glob(os.path.join(captures_path, "**/*.csv"), recursive=True)

    print(f"Found {len(pcap_files)} PCAP files and {len(csv_files)} CSV files")

    # Process PCAP files
    for pcap_path in pcap_files:
        filename = os.path.basename(pcap_path)

        # Extract label from filename (P0_, P1_, P2_, P3_)
        label = None
        for prefix in ["P0", "P1", "P2", "P3"]:
            if filename.upper().startswith(prefix):
                label = prefix
                break

        if not label:
            # Try to infer from directory name
            parent_dir = os.path.basename(os.path.dirname(pcap_path)).upper()
            for prefix in ["P0", "P1", "P2", "P3"]:
                if prefix in parent_dir:
                    label = prefix
                    break

        if not label:
            print(f"Could not determine label for {filename}, skipping")
            continue

        print(f"Processing {filename} as {label}...")
        flows = process_pcap_to_flows(pcap_path, label)
        all_flows.extend(flows)
        print(f"  Extracted {len(flows)} flows")

    # Process CSV files
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)

        label = None
        for prefix in ["P0", "P1", "P2", "P3"]:
            if filename.upper().startswith(prefix):
                label = prefix
                break

        if not label:
            parent_dir = os.path.basename(os.path.dirname(csv_path)).upper()
            for prefix in ["P0", "P1", "P2", "P3"]:
                if prefix in parent_dir:
                    label = prefix
                    break

        if not label:
            print(f"Could not determine label for {filename}, skipping")
            continue

        print(f"Processing {filename} as {label}...")
        flows = process_csv_export(csv_path, label)
        all_flows.extend(flows)
        print(f"  Extracted {len(flows)} flows")

    return pd.DataFrame(all_flows)


# %%
# Load all captures
print("Loading captures from:", CAPTURES_PATH)
print("=" * 60)

if os.path.exists(CAPTURES_PATH):
    df = load_all_captures(CAPTURES_PATH)
    print("=" * 60)
    print(f"\nTotal flows extracted: {len(df)}")
else:
    print(f"Captures path not found: {CAPTURES_PATH}")
    print("Creating sample data for demonstration...")

    # Create sample data for notebook demonstration
    np.random.seed(42)
    n_samples = 500

    sample_data = []
    for i in range(n_samples):
        label = np.random.choice(["P0", "P1", "P2", "P3"], p=[0.2, 0.4, 0.2, 0.2])

        # Generate realistic features based on label
        if label == "P3":  # Banking
            pkt_count = np.random.randint(10, 100)
            byte_count = pkt_count * np.random.randint(200, 800)
            duration = np.random.uniform(1, 30)
            pkt_mean = np.random.uniform(300, 600)
        elif label == "P2":  # Voice
            pkt_count = np.random.randint(100, 1000)
            byte_count = pkt_count * np.random.randint(150, 400)
            duration = np.random.uniform(30, 300)
            pkt_mean = np.random.uniform(150, 350)
        elif label == "P0":  # Bulk
            pkt_count = np.random.randint(500, 5000)
            byte_count = pkt_count * np.random.randint(1000, 1500)
            duration = np.random.uniform(10, 120)
            pkt_mean = np.random.uniform(1000, 1450)
        else:  # P1 Web
            pkt_count = np.random.randint(20, 500)
            byte_count = pkt_count * np.random.randint(300, 1000)
            duration = np.random.uniform(0.5, 60)
            pkt_mean = np.random.uniform(400, 900)

        sample_data.append(
            {
                "flow_id": i,
                "src_ip": f"192.168.1.{np.random.randint(1, 255)}",
                "dst_ip": f"10.0.0.{np.random.randint(1, 255)}",
                "src_port": np.random.randint(1024, 65535),
                "dst_port": np.random.choice([80, 443, 5060, 3478, 22, 21]),
                "protocol": np.random.choice([6, 17]),
                "packet_count": pkt_count,
                "byte_count": byte_count,
                "duration_sec": duration,
                "bytes_per_packet": byte_count / pkt_count,
                "packets_per_sec": pkt_count / duration,
                "bytes_per_sec": byte_count / duration,
                "pkt_len_min": int(pkt_mean * 0.3),
                "pkt_len_max": int(pkt_mean * 1.5),
                "pkt_len_mean": pkt_mean,
                "pkt_len_std": pkt_mean * 0.3,
                "iat_mean": duration / pkt_count,
                "iat_std": (duration / pkt_count) * 0.5,
                "sni_domain": "",
                "label": label,
                "source_file": "sample_data",
            }
        )

    df = pd.DataFrame(sample_data)
    print(f"Created {len(df)} sample flows for demonstration")

# %% [markdown]
# ## 2. Exploratory Data Analysis

# %%
# Basic statistics
print("Dataset Overview")
print("=" * 60)
print(f"Total flows: {len(df)}")
print(f"\nClass distribution:")
print(df["label"].value_counts())
print(f"\nClass percentages:")
print(df["label"].value_counts(normalize=True).round(3) * 100)

# %%
# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
ax1 = axes[0]
class_counts = df["label"].value_counts().sort_index()
colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
bars = ax1.bar(class_counts.index, class_counts.values, color=colors)
ax1.set_xlabel("Priority Class")
ax1.set_ylabel("Number of Flows")
ax1.set_title("Flow Distribution by Priority Class")
for bar, count in zip(bars, class_counts.values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(count),
        ha="center",
        va="bottom",
    )

# Pie chart
ax2 = axes[1]
ax2.pie(
    class_counts.values,
    labels=[f"{l}\n({PRIORITY_MAP[l]})" for l in class_counts.index],
    autopct="%1.1f%%",
    colors=colors,
    explode=[0.02] * len(class_counts),
)
ax2.set_title("Class Proportions")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "class_distribution.png"), dpi=150)
plt.show()

# %%
# Feature statistics by class
print("\nFeature Statistics by Class")
print("=" * 60)

feature_cols = [
    "packet_count",
    "byte_count",
    "duration_sec",
    "bytes_per_packet",
    "packets_per_sec",
    "bytes_per_sec",
    "pkt_len_mean",
    "pkt_len_std",
    "iat_mean",
    "iat_std",
]

stats_by_class = df.groupby("label")[feature_cols].agg(["mean", "std", "median"])
print(stats_by_class.round(2))

# %%
# Feature distributions by class
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    for label in sorted(df["label"].unique()):
        data = df[df["label"] == label][col]
        # Use log scale for better visualization if values vary a lot
        if data.max() / (data.min() + 1) > 100:
            data = np.log1p(data)
            ax.set_xlabel(f"log({col})")
        ax.hist(data, alpha=0.5, label=label, bins=30)
    ax.set_title(col)
    ax.legend()

# Remove unused subplots
for j in range(len(feature_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "feature_distributions.png"), dpi=150)
plt.show()

# %%
# Box plots for key features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

key_features = [
    "bytes_per_packet",
    "packets_per_sec",
    "bytes_per_sec",
    "pkt_len_std",
    "iat_mean",
    "duration_sec",
]

for i, col in enumerate(key_features):
    ax = axes[i]
    df.boxplot(column=col, by="label", ax=ax)
    ax.set_title(col)
    ax.set_xlabel("Priority Class")

plt.suptitle("Feature Distributions by Priority Class", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "feature_boxplots.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 3. Feature Correlation Analysis

# %%
# Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = df[feature_cols].corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    square=True,
    linewidths=0.5,
)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "correlation_matrix.png"), dpi=150)
plt.show()

# %%
# Identify highly correlated features
print("\nHighly Correlated Feature Pairs (|r| > 0.8):")
print("=" * 60)
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            print(
                f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}"
            )

# %% [markdown]
# ## 4. Feature Importance Preview

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Prepare data
X = df[feature_cols].fillna(0)
y = LabelEncoder().fit_transform(df["label"])

# Quick RF for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Plot feature importance
importance_df = pd.DataFrame(
    {"feature": feature_cols, "importance": rf.feature_importances_}
).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Preliminary)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "feature_importance.png"), dpi=150)
plt.show()

print("\nTop 5 Most Important Features:")
print(importance_df.tail(5).to_string(index=False))

# %% [markdown]
# ## 5. Save Processed Data

# %%
# Save processed flows to CSV
output_file = os.path.join(OUTPUT_PATH, "flows_processed.csv")
df.to_csv(output_file, index=False)
print(f"Saved processed flows to: {output_file}")

# Save summary statistics
summary = {
    "total_flows": len(df),
    "class_distribution": df["label"].value_counts().to_dict(),
    "feature_means": df[feature_cols].mean().to_dict(),
    "feature_stds": df[feature_cols].std().to_dict(),
    "capture_files": df["source_file"].nunique(),
    "processing_date": datetime.now().isoformat(),
}

import json

with open(os.path.join(OUTPUT_PATH, "data_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nData Summary:")
print(json.dumps(summary, indent=2))

# %% [markdown]
# ## 6. Next Steps
#
# After running this notebook:
#
# 1. Check `flows_processed.csv` for the extracted features
# 2. Review class distribution - ensure balanced or apply techniques
# 3. Proceed to `02_train_classifier.ipynb` for model training
#
# ### Data Quality Checklist
#
# - [ ] At least 30 flows per class
# - [ ] Multiple source files per class
# - [ ] No extreme class imbalance (worst class > 10% of total)
# - [ ] Features have reasonable distributions

# %%
# Data quality check
print("\n" + "=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)

min_flows_per_class = df["label"].value_counts().min()
max_flows_per_class = df["label"].value_counts().max()
imbalance_ratio = max_flows_per_class / min_flows_per_class

checks = [
    ("Min flows per class >= 30", min_flows_per_class >= 30, min_flows_per_class),
    ("Imbalance ratio < 5:1", imbalance_ratio < 5, f"{imbalance_ratio:.2f}:1"),
    (
        "All 4 classes present",
        len(df["label"].unique()) == 4,
        df["label"].unique().tolist(),
    ),
    (
        "No missing values",
        df[feature_cols].isna().sum().sum() == 0,
        df[feature_cols].isna().sum().sum(),
    ),
]

for check_name, passed, value in checks:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {check_name} (value: {value})")

print("\n" + "=" * 60)
if all(c[1] for c in checks):
    print("All checks passed! Ready for model training.")
else:
    print(
        "Some checks failed. Consider collecting more data or applying balancing techniques."
    )
