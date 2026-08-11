"""
EEG Eye State Classification - Test Script (30% Test Set)
Loads trained XGBoost model and evaluates on test dataset
"""
import pandas as pd
import numpy as np
import os
import joblib
import sys
import io
import warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score
)

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = os.path.dirname(__file__)  # final_programs
JEEVITHA_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "jeevitha model")
MODEL_PATH = os.path.join(BASE_DIR, "eeg_xgboost_full_14.pkl")
OUT = os.path.join(BASE_DIR, "eval_results")
os.makedirs(OUT, exist_ok=True)

TEST_FILE = os.path.join(JEEVITHA_MODEL_PATH, "full_test_14.csv")

# ==========================================================
# MAIN EVALUATION
# ==========================================================
print("\n" + "="*60)
print("  EEG EYE STATE DETECTION MODEL -- TEST (30%)")
print("="*60)

try:
    # Load model
    model = joblib.load(MODEL_PATH)
    print("[OK] EEG XGBoost model loaded")
    
    # Load test data
    df = pd.read_csv(TEST_FILE)
    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values
    
    print(f"[OK] Test dataset loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    
    print(f"\n--- EEG RESULTS ---")
    print(f"Accuracy     : {acc:.4f}")
    print(f"ROC-AUC      : {roc_auc:.4f}")
    print(f"F1-Score     : {f1:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"Sensitivity  : {sensitivity:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Eyes Open", "Eyes Closed"]))
    print("Confusion Matrix:\n", cm)
    
    # ROC Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("EEG ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eeg_roc.png"), dpi=150)
    plt.close()
    print(f"[OK] ROC plot saved -> {OUT}\\eeg_roc.png")
    
    # Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Eyes Open", "Eyes Closed"],
                yticklabels=["Eyes Open", "Eyes Closed"],
                ax=ax, cbar=True)
    ax.set_title("EEG Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eeg_confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix plot saved -> {OUT}\\eeg_confusion_matrix.png")
    
except Exception as e:
    print(f"[FAIL] EEG evaluation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[OK] EEG evaluation complete")
