# %% [markdown]
# # SDN Congestion Prediction - Time Series Model
#
# This notebook trains a congestion predictor that forecasts network link utilization
# to enable proactive traffic management in the SDN controller.
#
# ## Research Approach
#
# - Predict: Will link utilization exceed 80% in the next time window?
# - Features: Historical utilization, time-of-day patterns, rolling statistics
# - Models: RandomForest (baseline), Gradient Boosting, optional LSTM

# %%
# Install dependencies
# !pip install scikit-learn pandas numpy matplotlib seaborn joblib

# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for CLI
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    TimeSeriesSplit,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

import joblib

# %%
# Mount Google Drive (for Colab)
try:
    from google.colab import drive

    drive.mount("/content/drive")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# %%
# Configuration
if IN_COLAB:
    DATA_PATH = "/content/drive/MyDrive/SDN_Project/processed/link_timeseries.csv"
    OUTPUT_PATH = "/content/drive/MyDrive/SDN_Project/models"
else:
    DATA_PATH = "./data/processed/link_timeseries.csv"
    OUTPUT_PATH = "./ml/models"

os.makedirs(OUTPUT_PATH, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Congestion threshold
CONGESTION_THRESHOLD = (
    0.65  # 65% utilization (lowered for synthetic data to get more positive examples)
)

# Prediction horizon
PREDICTION_HORIZON = 1  # Predict next interval

# Feature window
LOOKBACK_WINDOWS = [1, 3, 5, 10]  # Number of past intervals to consider

# %% [markdown]
# ## 1. Load or Generate Data
#
# For congestion prediction, we need time-series data of link utilization.
# If you have Prometheus/OVS data, use that. Otherwise, we'll generate synthetic
# data that simulates realistic network patterns.


# %%
def generate_synthetic_link_data(n_days=7, interval_seconds=10):
    """
    Generate realistic synthetic link utilization data.

    Simulates:
    - Diurnal patterns (high during work hours)
    - Weekly patterns (lower on weekends)
    - Random bursts
    - Gradual ramps
    """
    samples_per_day = (24 * 3600) // interval_seconds
    total_samples = n_days * samples_per_day

    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=n_days),
        periods=total_samples,
        freq=f"{interval_seconds}S",
    )

    data = []

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        minute = ts.minute
        weekday = ts.weekday()
        is_weekend = weekday >= 5

        # Base utilization with diurnal pattern
        if 9 <= hour <= 17:  # Work hours
            base_util = 0.5 + 0.2 * np.sin((hour - 9) * np.pi / 8)
        elif 7 <= hour <= 9:  # Morning ramp-up
            base_util = 0.3 + 0.2 * (hour - 7) / 2
        elif 17 <= hour <= 19:  # Evening ramp-down
            base_util = 0.5 - 0.2 * (hour - 17) / 2
        else:  # Night
            base_util = 0.1 + 0.1 * np.random.random()

        # Weekend reduction
        if is_weekend:
            base_util *= 0.4

        # 9AM spike (login surge)
        if hour == 9 and minute < 30:
            base_util += 0.3 * np.random.random()

        # Random bursts (5% chance)
        if np.random.random() < 0.05:
            base_util += 0.2 * np.random.random()

        # Add noise
        noise = np.random.normal(0, 0.05)
        utilization = np.clip(base_util + noise, 0, 1)

        # Simulate bytes delta (based on 100 Mbps link)
        capacity_bps = 100_000_000
        bytes_delta = int(utilization * capacity_bps * interval_seconds / 8)

        data.append(
            {
                "timestamp": ts,
                "switch": "s1",
                "port": 3,
                "bytes_delta": bytes_delta,
                "utilization": utilization,
                "hour_of_day": hour,
                "minute_of_hour": minute,
                "day_of_week": weekday,
                "is_weekend": is_weekend,
            }
        )

    df = pd.DataFrame(data)

    # Add congestion label (1 if next interval exceeds threshold)
    df["next_utilization"] = df["utilization"].shift(-1)
    df["is_congested_next"] = (df["next_utilization"] > CONGESTION_THRESHOLD).astype(
        int
    )
    df = df.dropna()

    return df


# %%
# Load or generate data
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df)} records from {DATA_PATH}")
else:
    print(f"Data file not found: {DATA_PATH}")
    print("Generating synthetic data for demonstration...")
    df = generate_synthetic_link_data(
        n_days=7, interval_seconds=60
    )  # Faster: 7 days, 60s intervals
    print(f"Generated {len(df)} samples")

print(f"\nData range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Congestion rate: {df['is_congested_next'].mean():.2%}")

# %% [markdown]
# ## 2. Feature Engineering


# %%
def create_time_series_features(df, lookback_windows=[1, 3, 5, 10]):
    """
    Create time-series features for congestion prediction.

    Features:
    - Lagged utilization values
    - Rolling mean/std/max
    - Rate of change
    - Time-based features
    """
    df = df.copy()

    # Ensure sorted by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lagged features
    for lag in lookback_windows:
        df[f"util_lag_{lag}"] = df["utilization"].shift(lag)

    # Rolling statistics (use min_periods to avoid too many NaNs)
    for window in lookback_windows:
        df[f"util_rolling_mean_{window}"] = (
            df["utilization"].rolling(window, min_periods=1).mean()
        )
        df[f"util_rolling_std_{window}"] = (
            df["utilization"].rolling(window, min_periods=min(2, window)).std()
        )
        df[f"util_rolling_max_{window}"] = (
            df["utilization"].rolling(window, min_periods=1).max()
        )

    # Rate of change
    df["util_diff_1"] = df["utilization"].diff()
    df["util_diff_3"] = df["utilization"].diff(3)

    # Acceleration (second derivative)
    df["util_accel"] = df["util_diff_1"].diff()

    # Time features (cyclical encoding) - handle both timestamp and hour_of_day columns
    if "hour_of_day" in df.columns:
        hour = df["hour_of_day"]
    else:
        hour = pd.to_datetime(df["timestamp"]).dt.hour
        df["hour_of_day"] = hour

    if "day_of_week" in df.columns:
        day = df["day_of_week"]
    else:
        day = pd.to_datetime(df["timestamp"]).dt.dayofweek
        df["day_of_week"] = day

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * day / 7)
    df["day_cos"] = np.cos(2 * np.pi * day / 7)

    # Boolean flags
    df["is_work_hours"] = ((hour >= 9) & (hour <= 17)).astype(int)
    df["is_morning_peak"] = ((hour >= 8) & (hour <= 10)).astype(int)
    df["is_evening_peak"] = ((hour >= 16) & (hour <= 18)).astype(int)

    # Fill remaining NaN values with appropriate defaults
    df["util_diff_1"] = df["util_diff_1"].fillna(0)
    df["util_diff_3"] = df["util_diff_3"].fillna(0)
    df["util_accel"] = df["util_accel"].fillna(0)

    # For rolling std, fill NaN with 0 (single value has no std)
    for window in lookback_windows:
        df[f"util_rolling_std_{window}"] = df[f"util_rolling_std_{window}"].fillna(0)

    # Drop only the rows where lagged features are NaN (first max(lookback) rows)
    max_lookback = max(lookback_windows)
    df = df.iloc[max_lookback:].reset_index(drop=True)

    return df


# %%
# Create features
df_features = create_time_series_features(df, LOOKBACK_WINDOWS)
print(f"Samples after feature engineering: {len(df_features)}")

# Define feature columns
FEATURE_COLS = [
    col
    for col in df_features.columns
    if col
    not in [
        "timestamp",
        "switch",
        "port",
        "bytes_delta",
        "utilization",
        "next_utilization",
        "is_congested_next",
        "is_weekend",
        "day_of_week",
        "hour_of_day",
        "minute_of_hour",
    ]
]

print(f"\nFeatures ({len(FEATURE_COLS)}):")
for col in FEATURE_COLS:
    print(f"  - {col}")

# %% [markdown]
# ## 3. Exploratory Data Analysis

# %%
# Visualize utilization over time
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Full time series (sample for visibility)
sample_df = df_features.iloc[::10]  # Every 10th sample
axes[0].plot(sample_df["timestamp"], sample_df["utilization"], alpha=0.7)
axes[0].axhline(
    y=CONGESTION_THRESHOLD, color="r", linestyle="--", label="Congestion Threshold"
)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Utilization")
axes[0].set_title("Link Utilization Over Time")
axes[0].legend()

# Daily pattern
hourly_util = df_features.groupby("hour_of_day")["utilization"].agg(["mean", "std"])
axes[1].fill_between(
    hourly_util.index,
    hourly_util["mean"] - hourly_util["std"],
    hourly_util["mean"] + hourly_util["std"],
    alpha=0.3,
)
axes[1].plot(hourly_util.index, hourly_util["mean"], "o-", linewidth=2)
axes[1].axhline(
    y=CONGESTION_THRESHOLD, color="r", linestyle="--", label="Congestion Threshold"
)
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Utilization")
axes[1].set_title("Average Utilization by Hour")
axes[1].legend()
axes[1].set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "utilization_patterns.png"), dpi=150)
plt.show()

# %%
# Class balance
print("\nClass Distribution (Congestion Events):")
class_counts = df_features["is_congested_next"].value_counts()
total_samples = len(df_features)
if total_samples > 0:
    print(
        f"  Not congested (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0) / total_samples:.1%})"
    )
    print(
        f"  Congested (1):     {class_counts.get(1, 0)} ({class_counts.get(1, 0) / total_samples:.1%})"
    )
else:
    print("  ERROR: No samples after feature engineering!")
    print("  Check the create_time_series_features function.")

# %%
# Feature correlations with target
correlations = (
    df_features[FEATURE_COLS + ["is_congested_next"]]
    .corr()["is_congested_next"]
    .drop("is_congested_next")
)
correlations_sorted = correlations.abs().sort_values(ascending=False)

plt.figure(figsize=(10, 8))
colors = ["green" if c > 0 else "red" for c in correlations[correlations_sorted.index]]
plt.barh(
    correlations_sorted.index,
    correlations[correlations_sorted.index],
    color=colors,
    alpha=0.7,
)
plt.xlabel("Correlation with Congestion")
plt.title("Feature Correlations with Next-Interval Congestion")
plt.axvline(x=0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "feature_correlations_predictor.png"), dpi=150)
plt.show()

print("\nTop 10 Correlated Features:")
print(correlations_sorted.head(10))

# %% [markdown]
# ## 4. Model Training (Classification)
#
# We'll train a binary classifier: predict if congestion occurs in the next interval.

# %%
# Prepare data
X = df_features[FEATURE_COLS].values
y = df_features["is_congested_next"].values

# Time-series split (respect temporal order)
# Use last 20% as test set
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Training congestion rate: {y_train.mean():.2%}")
print(f"Test congestion rate: {y_test.mean():.2%}")

# %%
# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Model comparison
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    ),
}

print("Model Comparison (Time Series CV)")
print("=" * 60)

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring="f1")
    results[name] = {"mean": scores.mean(), "std": scores.std()}
    print(f"{name:25} F1: {scores.mean():.4f} (+/- {scores.std():.4f})")

# %%
# Select best model and tune hyperparameters
print("\nHyperparameter Tuning (Gradient Boosting)")
print("=" * 60)

gb_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "min_samples_split": [2, 5],
}

gb_base = GradientBoostingClassifier(random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    gb_base, gb_param_grid, cv=tscv, scoring="f1", n_jobs=-1, verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# %% [markdown]
# ## 5. Model Evaluation

# %%
# Final pipeline
final_pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", best_model)])

final_pipeline.fit(X_train, y_train)

# Predictions
y_pred = final_pipeline.predict(X_test)
y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

# %%
# Classification report
print("\nClassification Report (Test Set)")
print("=" * 60)
print(
    classification_report(y_test, y_pred, target_names=["No Congestion", "Congestion"])
)

# Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred_proba),
}

print("\nMetrics Summary:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# %%
# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0],
    xticklabels=["No Congestion", "Congestion"],
    yticklabels=["No Congestion", "Congestion"],
)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "predictor_evaluation.png"), dpi=150)
plt.show()

# %%
# Feature importance
if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    plt.xlabel("Feature Importance")
    plt.title("Congestion Predictor - Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "predictor_feature_importance.png"), dpi=150)
    plt.show()

    print("\nTop 5 Most Important Features:")
    print(importance_df.tail(5).to_string(index=False))

# %% [markdown]
# ## 6. Time Series Visualization

# %%
# Visualize predictions over time
test_df = df_features.iloc[split_idx:].copy()
test_df["predicted_congestion"] = y_pred
test_df["prediction_proba"] = y_pred_proba

# Sample for visualization
sample_size = min(500, len(test_df))
sample_df = test_df.iloc[:sample_size]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Utilization and predictions
ax1 = axes[0]
ax1.plot(
    sample_df["timestamp"], sample_df["utilization"], label="Utilization", alpha=0.8
)
ax1.axhline(
    y=CONGESTION_THRESHOLD, color="r", linestyle="--", alpha=0.7, label="Threshold"
)

# Mark actual congestion events
congested = sample_df[sample_df["is_congested_next"] == 1]
ax1.scatter(
    congested["timestamp"],
    congested["utilization"],
    color="red",
    s=20,
    alpha=0.5,
    label="Actual Congestion",
    zorder=5,
)

ax1.set_ylabel("Utilization")
ax1.set_title("Congestion Prediction Performance")
ax1.legend()

# Prediction probability
ax2 = axes[1]
ax2.plot(
    sample_df["timestamp"],
    sample_df["prediction_proba"],
    color="orange",
    alpha=0.8,
    label="Congestion Probability",
)
ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Decision Threshold")
ax2.set_xlabel("Time")
ax2.set_ylabel("Probability")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "prediction_timeseries.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 7. Save Model

# %%
# Save the final pipeline
model_path = os.path.join(OUTPUT_PATH, "congestion_predictor.pkl")
joblib.dump(final_pipeline, model_path)
print(f"Model saved to: {model_path}")

# Save feature names
feature_path = os.path.join(OUTPUT_PATH, "predictor_features.json")
with open(feature_path, "w") as f:
    json.dump(FEATURE_COLS, f)
print(f"Feature names saved to: {feature_path}")

# Save metadata
metadata = {
    "model_type": "GradientBoostingClassifier",
    "best_params": grid_search.best_params_,
    "cv_score": grid_search.best_score_,
    "test_metrics": metrics,
    "congestion_threshold": CONGESTION_THRESHOLD,
    "lookback_windows": LOOKBACK_WINDOWS,
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "training_date": datetime.now().isoformat(),
    "random_state": RANDOM_STATE,
}

metadata_path = os.path.join(OUTPUT_PATH, "predictor_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

# %% [markdown]
# ## 8. Integration Code

# %%
print("\n" + "=" * 60)
print("INTEGRATION CODE FOR ORCHESTRATOR")
print("=" * 60)

integration_code = '''
# Add this to orchestrator/stubs/predictor_stub.py

import joblib
import numpy as np
from collections import deque
from typing import Dict, Tuple

class MLCongestionPredictor:
    """ML-based congestion predictor for proactive traffic management."""
    
    def __init__(
        self, 
        model_path: str = "ml/models/congestion_predictor.pkl",
        lookback_windows: list = [1, 3, 5, 10]
    ):
        self.model = joblib.load(model_path)
        self.lookback_windows = lookback_windows
        self.history: Dict[str, deque] = {}  # link_id -> utilization history
        self.max_history = max(lookback_windows) + 1
    
    def update(self, link_id: str, utilization: float, hour: int, minute: int):
        """Update utilization history for a link."""
        if link_id not in self.history:
            self.history[link_id] = deque(maxlen=self.max_history)
        self.history[link_id].append({
            'utilization': utilization,
            'hour': hour,
            'minute': minute
        })
    
    def predict(self, link_id: str) -> Tuple[bool, float]:
        """
        Predict if congestion will occur in the next interval.
        
        Returns:
            (is_congested_predicted, probability)
        """
        if link_id not in self.history or len(self.history[link_id]) < max(self.lookback_windows):
            return False, 0.0
        
        features = self._extract_features(link_id)
        X = np.array([features])
        
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0][1]
        
        return bool(pred), float(proba)
    
    def _extract_features(self, link_id: str) -> list:
        """Extract features from utilization history."""
        history = list(self.history[link_id])
        utils = [h['utilization'] for h in history]
        current = history[-1]
        
        features = []
        
        # Lagged values
        for lag in self.lookback_windows:
            if len(utils) > lag:
                features.append(utils[-1-lag])
            else:
                features.append(utils[0])
        
        # Rolling statistics
        for window in self.lookback_windows:
            window_data = utils[-window:] if len(utils) >= window else utils
            features.extend([
                np.mean(window_data),
                np.std(window_data) if len(window_data) > 1 else 0,
                np.max(window_data)
            ])
        
        # Differences
        features.append(utils[-1] - utils[-2] if len(utils) > 1 else 0)
        features.append(utils[-1] - utils[-4] if len(utils) > 3 else 0)
        features.append((utils[-1] - utils[-2]) - (utils[-2] - utils[-3]) if len(utils) > 2 else 0)
        
        # Time features
        hour = current['hour']
        day = 0  # Assume weekday for simplicity
        features.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 7),
            np.cos(2 * np.pi * day / 7),
            int(9 <= hour <= 17),  # is_work_hours
            int(8 <= hour <= 10),  # is_morning_peak
            int(16 <= hour <= 18)  # is_evening_peak
        ])
        
        return features
'''

print(integration_code)

# %% [markdown]
# ## 9. Summary
#
# ### Congestion Prediction Results
#
# The model predicts with reasonable accuracy whether network congestion
# will occur in the next time interval, enabling proactive traffic management.
#
# ### Key Findings
#
# 1. **Most Predictive Features**: Rolling mean utilization, lagged values, time-of-day
# 2. **Best Model**: Gradient Boosting with tuned hyperparameters
# 3. **Use Case**: Proactive rerouting of P3 (Banking) traffic before congestion hits
#
# ### Files Generated
#
# - `congestion_predictor.pkl` - Trained prediction pipeline
# - `predictor_features.json` - Feature column names
# - `predictor_metadata.json` - Training metadata

# %%
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\nTest F1 Score: {metrics['f1']:.4f}")
print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"\nModel saved to: {model_path}")
print("\nReady for integration with SDN orchestrator!")
