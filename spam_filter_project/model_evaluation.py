import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc
)

# CONFIG
MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# LOAD MODEL AND TEST DATA
best_mnb = joblib.load(os.path.join(MODELS_DIR, "MNB_best.pkl"))
X_test_count = joblib.load(os.path.join(MODELS_DIR, "X_test_count.pkl"))
y_test = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))

# 1. STANDARD METRICS (Default Threshold = 0.5)
y_pred = best_mnb.predict(X_test_count)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1)
rec = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

metrics_df = pd.DataFrame([{
    "Model": "MultinomialNB",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1
}])

metrics_path = os.path.join(REPORTS_DIR, "MNB_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"[EVALUATION] Metrics saved to {metrics_path}")
print(metrics_df)

# 2. ADJUSTABLE RISK 
y_scores = best_mnb.predict_proba(X_test_count)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

TARGET_PRECISION = 0.95
THRESHOLD_FILE = os.path.join(MODELS_DIR, "high_precision_threshold.pkl")

threshold_indices = np.where(precision[:-1] >= TARGET_PRECISION)[0]

if threshold_indices.size > 0:
    best_idx = threshold_indices[-1]
    selected_threshold = thresholds[best_idx]
    selected_precision = precision[best_idx]
    selected_recall = recall[best_idx]

    joblib.dump(selected_threshold, THRESHOLD_FILE)
    print("\n[ADJUSTABLE RISK ANALYSIS]")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Target Precision: {TARGET_PRECISION}")
    print(f"Selected Threshold: {selected_threshold:.4f}")
    print(f"Precision @ threshold: {selected_precision:.4f}")
    print(f"Recall @ threshold: {selected_recall:.4f}")
    print(f"[EVALUATION] High-Precision Threshold saved to {THRESHOLD_FILE}")

    # Visualize Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'MNB (PR-AUC: {pr_auc:.4f})')
    plt.plot(selected_recall, selected_precision, 'ro', label=f'Low-Risk Threshold ({selected_threshold:.4f})')
    plt.xlabel('Recall (Spam Detection Rate)')
    plt.ylabel('Precision (Confidence in Spam Prediction)')
    plt.title('Precision-Recall Curve & Adjustable Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pr_path = os.path.join(REPORTS_DIR, "MNB_PR_Curve.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    print(f"[EVALUATION] Precision-Recall Curve saved to {pr_path}")

else:
    print("\n[WARNING] No threshold meets the required precision target.")
