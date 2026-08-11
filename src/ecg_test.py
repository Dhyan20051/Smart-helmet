import os
import numpy as np
import wfdb
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # → D:\Mini Project
DATASET_PATH = os.path.join(BASE_DIR, "mit-bih-arrhythmia-database-p-wave-annotations")
RECORDS = ["119", "122", "214", "223"]
FS = 360
WINDOW = FS
P_WAVE_WIDTH = int(0.12 * FS)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ecg_model_70.h5")

# =========================================================
# LOAD MODEL
# =========================================================
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded from:", MODEL_PATH)

# =========================================================
# HELPER
# =========================================================
def is_far_from_pwaves(idx, pwave_samples, margin):
    return np.all(np.abs(pwave_samples - idx) > margin)

# =========================================================
# LOAD TEST DATA
# =========================================================
X, y = [], []

print("Loading ECG test data...")

for rec in RECORDS:
    print(f"Processing record: {rec}")

    record = wfdb.rdrecord(os.path.join(DATASET_PATH, rec))
    ann = wfdb.rdann(os.path.join(DATASET_PATH, rec), "pwave")

    ecg = record.p_signal[:, 0]
    pwave_samples = ann.sample

    # ---------- Positive ----------
    for s in pwave_samples:
        if s - WINDOW//2 < 0 or s + WINDOW//2 >= len(ecg):
            continue
        X.append(ecg[s - WINDOW//2 : s + WINDOW//2])
        y.append(1)

    # ---------- Negative ----------
    neg_count = 0
    required_neg = len(pwave_samples)
    attempts = 0
    MAX_ATTEMPTS = required_neg * 10

    while neg_count < required_neg and attempts < MAX_ATTEMPTS:
        idx = np.random.randint(WINDOW, len(ecg) - WINDOW)
        attempts += 1

        if is_far_from_pwaves(idx, pwave_samples, margin=P_WAVE_WIDTH):
            X.append(ecg[idx - WINDOW//2 : idx + WINDOW//2])
            y.append(0)
            neg_count += 1

X = np.array(X)
y = np.array(y)

print("Total windows:", X.shape)
print("Label distribution:", np.unique(y, return_counts=True))

# =========================================================
# NORMALIZATION
# =========================================================
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
X = X[..., np.newaxis]

# =========================================================
# USE ONLY 30% FOR TESTING
# =========================================================
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

print("Testing samples:", X_test.shape)

# =========================================================
# PREDICTION
# =========================================================
y_scores = model.predict(X_test, verbose=0).ravel()
y_pred = (y_scores > 0.5).astype(int)

# =========================================================
# METRICS
# =========================================================
test_acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print("\n================ TEST RESULTS (30%) ================")
print("Testing Accuracy:", test_acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", cm)
print(f"Sensitivity (P-wave): {sensitivity:.3f}")
print(f"Specificity (Non P-wave): {specificity:.3f}")

# =========================================================
# ROC-AUC
# =========================================================
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

print("ROC-AUC:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - ECG P-wave Detection (Unseen Test Data)")
plt.legend()
plt.show()
