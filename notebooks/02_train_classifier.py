# %% [markdown]
# # SDN Traffic Classification - Model Training
#
# This notebook trains a traffic priority classifier (P0-P3) using the processed
# flow features from captured network traffic.
#
# ## Research Approach
#
# We follow a rigorous ML pipeline:
# 1. Stratified train/test split (80/20)
# 2. Feature scaling with StandardScaler
# 3. Hyperparameter tuning with GridSearchCV
# 4. 5-fold cross-validation
# 5. Multiple model comparison
# 6. Statistical significance testing

# %%
# Install dependencies
# !pip install scikit-learn pandas numpy matplotlib seaborn joblib imbalanced-learn

# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Scikit-learn
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# For imbalanced data
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("imbalanced-learn not available, will use class weights instead")

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
    DATA_PATH = "/content/drive/MyDrive/SDN_Project/processed/flows_processed.csv"
    OUTPUT_PATH = "/content/drive/MyDrive/SDN_Project/models"
else:
    DATA_PATH = "./data/processed/flows_processed.csv"
    OUTPUT_PATH = "./ml/models"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# %% [markdown]
# ## 1. Load and Prepare Data

# %%
# Load processed data
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} flows from {DATA_PATH}")
else:
    print(f"Data file not found: {DATA_PATH}")
    print("Please run 01_data_exploration.ipynb first")
    print("\nCreating sample data for demonstration...")

    # Sample data for demonstration
    np.random.seed(42)
    n = 500
    data = []
    for i in range(n):
        label = np.random.choice(["P0", "P1", "P2", "P3"], p=[0.2, 0.4, 0.2, 0.2])

        if label == "P3":
            pkt_count = np.random.randint(10, 100)
            pkt_mean = np.random.uniform(300, 600)
            iat_mean = np.random.uniform(0.01, 0.1)
        elif label == "P2":
            pkt_count = np.random.randint(100, 1000)
            pkt_mean = np.random.uniform(150, 350)
            iat_mean = np.random.uniform(0.015, 0.04)
        elif label == "P0":
            pkt_count = np.random.randint(500, 5000)
            pkt_mean = np.random.uniform(1000, 1450)
            iat_mean = np.random.uniform(0.0001, 0.005)
        else:
            pkt_count = np.random.randint(20, 500)
            pkt_mean = np.random.uniform(400, 900)
            iat_mean = np.random.uniform(0.005, 0.1)

        duration = pkt_count * iat_mean * np.random.uniform(0.8, 1.2)
        byte_count = int(pkt_count * pkt_mean)

        data.append(
            {
                "packet_count": pkt_count,
                "byte_count": byte_count,
                "duration_sec": duration,
                "bytes_per_packet": pkt_mean,
                "packets_per_sec": pkt_count / max(duration, 0.001),
                "bytes_per_sec": byte_count / max(duration, 0.001),
                "pkt_len_min": int(pkt_mean * 0.3),
                "pkt_len_max": int(min(pkt_mean * 1.5, 1500)),
                "pkt_len_mean": pkt_mean,
                "pkt_len_std": pkt_mean * 0.3,
                "iat_mean": iat_mean,
                "iat_std": iat_mean * 0.5,
                "label": label,
            }
        )

    df = pd.DataFrame(data)
    print(f"Created {len(df)} sample flows")

# %%
# Define features
FEATURE_COLS = [
    "packet_count",
    "byte_count",
    "duration_sec",
    "bytes_per_packet",
    "packets_per_sec",
    "bytes_per_sec",
    "pkt_len_min",
    "pkt_len_max",
    "pkt_len_mean",
    "pkt_len_std",
    "iat_mean",
    "iat_std",
]

# Verify columns exist
available_features = [col for col in FEATURE_COLS if col in df.columns]
print(f"Using {len(available_features)} features: {available_features}")

# Prepare X and y
X = df[available_features].fillna(0).values
y = df["label"].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

print(f"\nClasses: {list(class_names)}")
print(f"Class distribution: {dict(zip(class_names, np.bincount(y_encoded)))}")

# %%
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# %% [markdown]
# ## 2. Model Selection and Comparison

# %%
# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
    ),
}

# %%
# Cross-validation comparison
print("Model Comparison (5-Fold CV)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1_macro")
    results[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
    print(f"{name:25} F1-macro: {scores.mean():.4f} (+/- {scores.std():.4f})")

# %%
# Visualize model comparison
plt.figure(figsize=(12, 6))

model_names = list(results.keys())
means = [results[m]["mean"] for m in model_names]
stds = [results[m]["std"] for m in model_names]

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
bars = plt.bar(
    model_names, means, yerr=stds, capsize=5, color=colors, edgecolor="black"
)

plt.ylabel("F1-Macro Score")
plt.title("Model Comparison (5-Fold Cross-Validation)")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)

# Add value labels
for bar, mean in zip(bars, means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{mean:.3f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "model_comparison.png"), dpi=150)
plt.show()

# %%
# Select best model
best_model_name = max(results, key=lambda x: results[x]["mean"])
print(f"\nBest performing model: {best_model_name}")
print(f"F1-macro: {results[best_model_name]['mean']:.4f}")

# %% [markdown]
# ## 3. Hyperparameter Tuning (Random Forest)

# %%
# We'll focus on Random Forest as it typically performs well for this task
print("\nHyperparameter Tuning for Random Forest")
print("=" * 60)

rf_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

# For faster tuning, use a subset of parameters
rf_param_grid_fast = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

rf_base = RandomForestClassifier(
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)

grid_search = GridSearchCV(
    rf_base, rf_param_grid_fast, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# %%
# Use the best estimator
best_rf = grid_search.best_estimator_

# %% [markdown]
# ## 4. Final Model Training and Evaluation

# %%
# Create final pipeline
final_pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", best_rf)])

# Fit on full training data
final_pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = final_pipeline.predict(X_test)
y_pred_proba = final_pipeline.predict_proba(X_test)

# %%
# Classification report
print("\nClassification Report (Test Set)")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=class_names))

# %%
# Metrics summary
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_macro": f1_score(y_test, y_pred, average="macro"),
    "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    "precision_macro": precision_score(y_test, y_pred, average="macro"),
    "recall_macro": recall_score(y_test, y_pred, average="macro"),
}

print("\nMetrics Summary:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# %%
# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "confusion_matrix.png"), dpi=150)
plt.show()

# Normalized confusion matrix
plt.figure(figsize=(10, 8))
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2%",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "confusion_matrix_normalized.png"), dpi=150)
plt.show()

# %%
# Feature importance
feature_importance = pd.DataFrame(
    {"feature": available_features, "importance": best_rf.feature_importances_}
).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(
    feature_importance["feature"], feature_importance["importance"], color="steelblue"
)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "feature_importance_final.png"), dpi=150)
plt.show()

print("\nTop 5 Most Important Features:")
print(feature_importance.tail(5).to_string(index=False))

# %% [markdown]
# ## 5. Learning Curves

# %%
# Learning curve analysis
train_sizes, train_scores, test_scores = learning_curve(
    best_rf,
    X_train_scaled,
    y_train,
    cv=cv,
    scoring="f1_macro",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(
    train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue"
)
plt.fill_between(
    train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange"
)
plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
plt.plot(train_sizes, test_mean, "o-", color="orange", label="Cross-validation score")
plt.xlabel("Training Set Size")
plt.ylabel("F1-Macro Score")
plt.title("Learning Curves")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "learning_curves.png"), dpi=150)
plt.show()

# Interpret learning curves
gap = train_mean[-1] - test_mean[-1]
if gap > 0.1:
    print("\nNote: Large gap between training and CV scores suggests overfitting.")
    print("Consider: more regularization, fewer features, or more training data.")
elif test_mean[-1] < 0.7:
    print("\nNote: Both scores are relatively low, suggesting underfitting.")
    print("Consider: more complex model or additional features.")
else:
    print("\nNote: Good convergence. Model appears well-fitted.")

# %% [markdown]
# ## 6. Save Model and Artifacts

# %%
# Save the final pipeline
model_path = os.path.join(OUTPUT_PATH, "traffic_classifier.pkl")
joblib.dump(final_pipeline, model_path)
print(f"Model saved to: {model_path}")

# Save label encoder
encoder_path = os.path.join(OUTPUT_PATH, "label_encoder.pkl")
joblib.dump(label_encoder, encoder_path)
print(f"Label encoder saved to: {encoder_path}")

# Save feature names
feature_path = os.path.join(OUTPUT_PATH, "feature_names.json")
with open(feature_path, "w") as f:
    json.dump(available_features, f)
print(f"Feature names saved to: {feature_path}")

# Save training metadata
metadata = {
    "model_type": "RandomForestClassifier",
    "best_params": grid_search.best_params_,
    "cv_score": grid_search.best_score_,
    "test_metrics": metrics,
    "feature_importance": feature_importance.set_index("feature")[
        "importance"
    ].to_dict(),
    "class_names": list(class_names),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "training_date": datetime.now().isoformat(),
    "random_state": RANDOM_STATE,
}

metadata_path = os.path.join(OUTPUT_PATH, "model_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

# %% [markdown]
# ## 7. Model Usage Example

# %%
# Example: How to use the trained model
print("\n" + "=" * 60)
print("MODEL USAGE EXAMPLE")
print("=" * 60)

# Load model (as would be done in production)
loaded_pipeline = joblib.load(model_path)
loaded_encoder = joblib.load(encoder_path)

# Example flow
example_flow = {
    "packet_count": 150,
    "byte_count": 45000,
    "duration_sec": 2.5,
    "bytes_per_packet": 300,
    "packets_per_sec": 60,
    "bytes_per_sec": 18000,
    "pkt_len_min": 64,
    "pkt_len_max": 1200,
    "pkt_len_mean": 300,
    "pkt_len_std": 150,
    "iat_mean": 0.017,
    "iat_std": 0.008,
}

# Prepare features
X_example = np.array([[example_flow[f] for f in available_features]])

# Predict
pred_encoded = loaded_pipeline.predict(X_example)
pred_proba = loaded_pipeline.predict_proba(X_example)
pred_label = loaded_encoder.inverse_transform(pred_encoded)[0]

print(f"\nInput flow features:")
for k, v in example_flow.items():
    print(f"  {k}: {v}")

print(f"\nPredicted class: {pred_label}")
print(f"\nClass probabilities:")
for cls, prob in zip(loaded_encoder.classes_, pred_proba[0]):
    print(f"  {cls}: {prob:.3f}")

# %%
# Integration code for SDN orchestrator
print("\n" + "=" * 60)
print("INTEGRATION CODE FOR ORCHESTRATOR")
print("=" * 60)

integration_code = '''
# Add this to orchestrator/stubs/classifier_stub.py

import joblib
import numpy as np
from pathlib import Path

class MLClassifier:
    """ML-based traffic classifier for priority assignment."""
    
    FEATURE_COLS = [
        'packet_count', 'byte_count', 'duration_sec',
        'bytes_per_packet', 'packets_per_sec', 'bytes_per_sec',
        'pkt_len_min', 'pkt_len_max', 'pkt_len_mean', 'pkt_len_std',
        'iat_mean', 'iat_std'
    ]
    
    def __init__(self, model_path: str = "ml/models/traffic_classifier.pkl"):
        self.model = joblib.load(model_path)
        encoder_path = Path(model_path).parent / "label_encoder.pkl"
        self.encoder = joblib.load(encoder_path)
    
    def classify(self, flow: dict) -> str:
        """Classify a flow and return priority label (P0-P3)."""
        features = [flow.get(f, 0) for f in self.FEATURE_COLS]
        X = np.array([features])
        
        pred = self.model.predict(X)
        return self.encoder.inverse_transform(pred)[0]
    
    def classify_with_confidence(self, flow: dict) -> tuple:
        """Return (label, confidence) tuple."""
        features = [flow.get(f, 0) for f in self.FEATURE_COLS]
        X = np.array([features])
        
        pred = self.model.predict(X)
        proba = self.model.predict_proba(X)
        
        label = self.encoder.inverse_transform(pred)[0]
        confidence = proba[0].max()
        
        return label, confidence
'''

print(integration_code)

# %% [markdown]
# ## 8. Summary
#
# ### Training Results
#
# | Metric | Value |
# |--------|-------|
# | Accuracy | {accuracy:.4f} |
# | F1-Macro | {f1_macro:.4f} |
# | F1-Weighted | {f1_weighted:.4f} |
#
# ### Files Generated
#
# - `traffic_classifier.pkl` - Trained model pipeline
# - `label_encoder.pkl` - Label encoder for class names
# - `feature_names.json` - List of feature columns
# - `model_metadata.json` - Training metadata and metrics
#
# ### Next Steps
#
# 1. Integrate model with orchestrator
# 2. Run `03_train_predictor.ipynb` for congestion prediction
# 3. Test end-to-end in SDN environment

# %%
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Final Test F1-Macro: {metrics['f1_macro']:.4f}")
print(f"\nModel saved to: {model_path}")
print("\nReady for integration with SDN orchestrator!")
