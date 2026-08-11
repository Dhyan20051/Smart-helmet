import os
import numpy as np
import pandas as pd
import joblib

from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # → D:\Mini Project
DATASET_PATH = os.path.join(BASE_DIR, "MobiFall_Dataset_v2.0")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "imu_mobifall_crash_model_new.pkl")

FS = 100                # Sampling frequency
WINDOW_SEC = 1.5        # 1.5 seconds
WINDOW = int(FS * WINDOW_SEC)
STEP = WINDOW // 2

# =====================================================
# LOW-PASS FILTER
# =====================================================

def lowpass(signal, fs=100, cutoff=10):
    b, a = butter(2, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, signal)

# =====================================================
# FEATURE EXTRACTION (ROBUST)
# =====================================================

def extract_features(ax, ay, az):
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    jerk = np.diff(mag)

    return [
        np.mean(mag),
        np.std(mag),
        np.max(mag),
        np.min(mag),
        np.sum(mag ** 2),           # energy
        np.mean(np.abs(jerk)),
        np.max(np.abs(jerk)),
        np.mean(ax),
        np.mean(ay),
        np.mean(az),
        np.var(mag),
        np.sqrt(np.mean(mag ** 2))  # RMS
    ]

# =====================================================
# LOAD FILE LIST (FILE-LEVEL SPLIT)
# =====================================================

def collect_files(base_path):
    files = []
    labels = []

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

files, file_labels = collect_files(DATASET_PATH)

print("Total IMU files:", len(files))
print("FALL files:", sum(file_labels))
print("ADL  files:", len(file_labels) - sum(file_labels))

# =====================================================
# 70% TRAIN FILES / 30% TEST FILES
# =====================================================

train_files, test_files, train_labels, test_labels = train_test_split(
    files,
    file_labels,
    train_size=0.70,
    stratify=file_labels,
    random_state=42
)

print("\nTrain files:", len(train_files))
print("Test files :", len(test_files))

# =====================================================
# LOAD + PROCESS FILES
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
            fx = extract_features(
                ax[i:i+WINDOW],
                ay[i:i+WINDOW],
                az[i:i+WINDOW]
            )
            X.append(fx)
            y.append(label)

    return np.array(X), np.array(y)

print("\nExtracting TRAIN features...")
X_train, y_train = process_files(train_files, train_labels)

print("Extracting TEST features...")
X_test, y_test = process_files(test_files, test_labels)

print("\nTrain windows:", X_train.shape)
print("Test windows :", X_test.shape)

# =====================================================
# TRAIN MODEL (ONLY ON 70%)
# =====================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining IMU model...")
model.fit(X_train, y_train)

# =====================================================
# TRAIN METRICS (FOR REFERENCE ONLY)
# =====================================================

train_pred = model.predict(X_train)
train_prob = model.predict_proba(X_train)[:, 1]

train_acc = accuracy_score(y_train, train_pred)
train_auc = roc_auc_score(y_train, train_prob)

print("\n===== TRAIN PERFORMANCE =====")
print("Train Accuracy:", round(train_acc, 4))
print("Train ROC-AUC :", round(train_auc, 4))

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(model, MODEL_SAVE_PATH)
print("\nModel saved as:", MODEL_SAVE_PATH)
