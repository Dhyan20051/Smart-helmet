import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # → D:\Mini Project
DATASET_PATH = os.path.join(BASE_DIR, "MobiFall_Dataset_v2.0")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "imu_mobifall_crash_model_new.pkl")

FS = 50
WINDOW_SEC = 1.5
WINDOW = int(FS * WINDOW_SEC)
STEP = WINDOW // 2

# =====================================================
# LOWPASS FILTER (SAME AS TRAINING)
# =====================================================

def lowpass(signal, fs=50, cutoff=10):
    b, a = butter(2, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, signal)

# =====================================================
# FEATURE EXTRACTION (IDENTICAL)
# =====================================================

def extract_features(ax, ay, az):
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    jerk = np.diff(mag)

    return [
        np.mean(mag),
        np.std(mag),
        np.max(mag),
        np.min(mag),
        np.sum(mag ** 2),
        np.mean(np.abs(jerk)),
        np.max(np.abs(jerk)),
        np.mean(ax),
        np.mean(ay),
        np.mean(az),
        np.var(mag),
        np.sqrt(np.mean(mag ** 2))
    ]

# =====================================================
# COLLECT FILES (FILE-LEVEL SPLIT)
# =====================================================

def collect_files(base_path):
    files, labels = [], []

    for root, _, filenames in os.walk(base_path):
        for f in filenames:
            if not f.endswith(".txt"):
                continue

            full_path = os.path.join(root, f)

            if "FALLS" in root.upper():
                files.append(full_path)
                labels.append(1)
            elif "ADL" in root.upper():
                files.append(full_path)
                labels.append(0)

    return files, labels

files, labels = collect_files(DATASET_PATH)

# SAME SPLIT AS TRAINING (RANDOM_STATE FIXED)
_, test_files, _, test_labels = train_test_split(
    files,
    labels,
    train_size=0.70,
    stratify=labels,
    random_state=42
)

print("Test files:", len(test_files))

# =====================================================
# LOAD TEST DATA
# =====================================================

def process_files(file_list, label_list):
    X, y = [], []

    for file_path, label in zip(file_list, label_list):
        rows = []

        with open(file_path, "r", errors="ignore") as f:
            for line in f:
                parts = line.replace(",", " ").split()
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except:
                        continue
                if len(nums) >= 3:
                    rows.append(nums[-3:])

        if len(rows) < WINDOW:
            continue

        data = np.array(rows)
        ax, ay, az = data[:, 0], data[:, 1], data[:, 2]

        ax = lowpass(ax)
        ay = lowpass(ay)
        az = lowpass(az)

        for i in range(0, len(ax) - WINDOW, STEP):
            X.append(extract_features(
                ax[i:i+WINDOW],
                ay[i:i+WINDOW],
                az[i:i+WINDOW]
            ))
            y.append(label)

    return np.array(X), np.array(y)

print("Extracting TEST features...")
X_test, y_test = process_files(test_files, test_labels)

print("Test windows:", X_test.shape)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(MODEL_PATH)
print("IMU model loaded from:", MODEL_PATH)

# =====================================================
# TEST PREDICTION
# =====================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)

print("\n================ IMU TEST RESULTS (30%) ================")
print("Test Accuracy:", round(test_acc, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# =====================================================
# ROC-AUC
# =====================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f"ROC-AUC: {round(roc_auc, 4)}")
print(f"Sensitivity (Fall): {sensitivity:.4f}")
print(f"Specificity (ADL) : {specificity:.4f}")

# =====================================================
# PLOTS
# =====================================================

# ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("IMU Crash Detection ROC Curve (Test Data)")
plt.legend()
plt.grid()
plt.show()

# Confusion Matrix
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("IMU Confusion Matrix (Test Data)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xticks([0, 1], ["ADL", "FALL"])
plt.yticks([0, 1], ["ADL", "FALL"])
plt.show()
