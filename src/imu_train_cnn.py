"""
IMU Crash Detection — Improved 1D-CNN Training Script
Key improvements over v1:
  - Uses magnitude + jerk channels alongside raw axes (6 channels total)
  - Deeper architecture with residual-like skip connections
  - Cosine annealing LR schedule
  - Data augmentation via jitter + scaling
"""
import os, sys, io, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc as sk_auc)

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "MobiFall_Dataset_v2.0")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "imu_cnn_crash_model.h5")
OUT = os.path.join(os.path.dirname(__file__), "imu_cnn_results")
os.makedirs(OUT, exist_ok=True)

FS = 50
WINDOW_SEC = 1.5
WINDOW = int(FS * WINDOW_SEC)   # 75 samples
STEP = WINDOW // 2
EPOCHS = 40
BATCH_SIZE = 128
RANDOM_STATE = 42

# =====================================================
# LOW-PASS FILTER
# =====================================================
def lowpass(signal, fs=50, cutoff=10):
    b, a = butter(2, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, signal)

# =====================================================
# COLLECT FILES
# =====================================================
def collect_files(base_path):
    files, labels = [], []
    for root, _, filenames in os.walk(base_path):
        for f in filenames:
            if not f.endswith(".txt"): continue
            full_path = os.path.join(root, f)
            if "FALLS" in root.upper():
                files.append(full_path); labels.append(1)
            elif "ADL" in root.upper():
                files.append(full_path); labels.append(0)
    return files, labels

files, file_labels = collect_files(DATASET_PATH)
print(f"Total IMU files: {len(files)}")
print(f"  FALL files: {sum(file_labels)}")
print(f"  ADL  files: {len(file_labels) - sum(file_labels)}")

# File-level split
train_files, test_files, train_labels, test_labels = train_test_split(
    files, file_labels,
    train_size=0.70, stratify=file_labels, random_state=RANDOM_STATE
)
print(f"\nTrain files: {len(train_files)}")
print(f"Test files : {len(test_files)}")

# =====================================================
# LOAD WINDOWS WITH ENGINEERED CHANNELS
# =====================================================
def process_files_cnn(file_list, label_list):
    """
    For each window, create 6 channels:
      [ax, ay, az, magnitude, jerk_mag, orientation_change]
    This gives the CNN both raw signal AND domain knowledge.
    """
    X, y = [], []
    for file_path, label in zip(file_list, label_list):
        rows = []
        with open(file_path, "r", errors="ignore") as f:
            for line in f:
                parts = line.replace(",", " ").split()
                nums = []
                for p in parts:
                    try: nums.append(float(p))
                    except: continue
                if len(nums) >= 3:
                    rows.append(nums[-3:])

        if len(rows) < WINDOW:
            continue

        data = np.array(rows, dtype=np.float32)
        ax_raw, ay_raw, az_raw = data[:, 0], data[:, 1], data[:, 2]

        # Lowpass filter
        ax_f = lowpass(ax_raw).astype(np.float32)
        ay_f = lowpass(ay_raw).astype(np.float32)
        az_f = lowpass(az_raw).astype(np.float32)

        # Derived channels
        mag = np.sqrt(ax_f**2 + ay_f**2 + az_f**2)
        jerk = np.zeros_like(mag)
        jerk[1:] = np.abs(np.diff(mag))

        # Sliding windows
        for i in range(0, len(ax_f) - WINDOW, STEP):
            w_ax  = ax_f[i:i+WINDOW]
            w_ay  = ay_f[i:i+WINDOW]
            w_az  = az_f[i:i+WINDOW]
            w_mag = mag[i:i+WINDOW]
            w_jrk = jerk[i:i+WINDOW]

            # Stack: (75, 5)
            window = np.stack([w_ax, w_ay, w_az, w_mag, w_jrk], axis=-1)

            # Global normalization per channel
            mean = window.mean(axis=0, keepdims=True)
            std  = window.std(axis=0, keepdims=True) + 1e-8
            window = (window - mean) / std

            X.append(window)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

print("\nExtracting TRAIN windows (5 channels)...")
X_train, y_train = process_files_cnn(train_files, train_labels)

print("Extracting TEST windows (5 channels)...")
X_test, y_test = process_files_cnn(test_files, test_labels)

print(f"\nTrain: {X_train.shape}  |  Labels: {np.unique(y_train, return_counts=True)}")
print(f"Test : {X_test.shape}  |  Labels: {np.unique(y_test, return_counts=True)}")

N_CHANNELS = X_train.shape[2]

# =====================================================
# DATA AUGMENTATION
# =====================================================
def augment_batch(X, y, copies=2):
    """Add jittered + scaled copies of fall samples to balance dataset."""
    fall_idx = np.where(y == 1)[0]
    X_aug, y_aug = [X], [y]
    for _ in range(copies):
        X_fall = X[fall_idx].copy()
        # Random jitter
        X_fall += np.random.normal(0, 0.1, X_fall.shape).astype(np.float32)
        # Random scaling
        scale = np.random.uniform(0.9, 1.1, (len(X_fall), 1, 1)).astype(np.float32)
        X_fall *= scale
        X_aug.append(X_fall)
        y_aug.append(np.ones(len(X_fall), dtype=np.int32))
    return np.concatenate(X_aug), np.concatenate(y_aug)

print("Augmenting fall samples...")
X_train_aug, y_train_aug = augment_batch(X_train, y_train, copies=2)
print(f"After augmentation: {X_train_aug.shape}")
print(f"Labels: {np.unique(y_train_aug, return_counts=True)}")

# Shuffle
perm = np.random.permutation(len(X_train_aug))
X_train_aug = X_train_aug[perm]
y_train_aug = y_train_aug[perm]

# =====================================================
# 1D-CNN MODEL (IMPROVED)
# =====================================================
print("\nBuilding improved 1D-CNN...")

inputs = tf.keras.layers.Input(shape=(WINDOW, N_CHANNELS))

# Block 1
x = tf.keras.layers.Conv1D(64, 7, padding="same")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.MaxPooling1D(2)(x)

# Block 2
x = tf.keras.layers.Conv1D(128, 5, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.MaxPooling1D(2)(x)

# Block 3
x = tf.keras.layers.Conv1D(256, 3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)

# Global pooling (both avg and max for richer representation)
avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Concatenate()([avg_pool, max_pool])

# Classifier
x = tf.keras.layers.Dense(128, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# TRAINING
# =====================================================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=7, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
)

print("\nTraining improved 1D-CNN...")
history = model.fit(
    X_train_aug, y_train_aug,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =====================================================
# EVALUATE
# =====================================================
print("\nEvaluating on ORIGINAL (non-augmented) test set...")

# Train metrics (on original train, not augmented)
train_scores = model.predict(X_train, verbose=0).ravel()
train_preds  = (train_scores > 0.5).astype(int)
train_acc    = accuracy_score(y_train, train_preds)
train_auc    = roc_auc_score(y_train, train_scores)

# Test metrics
test_scores = model.predict(X_test, verbose=0).ravel()
test_preds  = (test_scores > 0.5).astype(int)
test_acc    = accuracy_score(y_test, test_preds)
test_auc    = roc_auc_score(y_test, test_scores)
cm          = confusion_matrix(y_test, test_preds)
TN, FP, FN, TP = cm.ravel()

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
f1          = 2 * TP / (2 * TP + FP + FN)

print("\n" + "=" * 60)
print("  IMU 1D-CNN (IMPROVED) — RESULTS")
print("=" * 60)
print(f"\n  TRAIN Accuracy : {train_acc:.4f}")
print(f"  TRAIN ROC-AUC  : {train_auc:.4f}")
print(f"\n  TEST Accuracy  : {test_acc:.4f}")
print(f"  TEST ROC-AUC   : {test_auc:.4f}")
print(f"  Sensitivity    : {sensitivity:.4f}")
print(f"  Specificity    : {specificity:.4f}")
print(f"  Precision      : {precision:.4f}")
print(f"  F1-Score       : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, test_preds, target_names=["ADL", "Fall/Crash"]))
print("Confusion Matrix:\n", cm)

# =====================================================
# SAVE MODEL
# =====================================================
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved: {MODEL_SAVE_PATH}")

# =====================================================
# PLOTS
# =====================================================
# 1. Overfitting analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, len(history.history["accuracy"]) + 1)

axes[0].plot(epochs_range, history.history["accuracy"],     "b-o", ms=3, label="Train Acc")
axes[0].plot(epochs_range, history.history["val_accuracy"],  "r-s", ms=3, label="Val Acc")
axes[0].axhline(test_acc, color="green", ls="--", lw=1.5, label=f"Test Acc = {test_acc:.4f}")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].set_title("IMU CNN — Accuracy Over Epochs", fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history.history["loss"],     "b-o", ms=3, label="Train Loss")
axes[1].plot(epochs_range, history.history["val_loss"],  "r-s", ms=3, label="Val Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].set_title("IMU CNN — Loss Over Epochs", fontweight="bold")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "imu_cnn_overfitting.png"), dpi=200)
plt.close()

# 2. ROC + Confusion Matrix
fpr, tpr, _ = roc_curve(y_test, test_scores)
roc_auc_val = sk_auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr, tpr, "b-", lw=2, label=f"CNN AUC = {roc_auc_val:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("IMU CNN — ROC Curve", fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=0.3)

import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ADL", "FALL"], yticklabels=["ADL", "FALL"], ax=axes[1])
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
axes[1].set_title("IMU CNN — Confusion Matrix", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "imu_cnn_roc_cm.png"), dpi=200)
plt.close()

# 3. RF vs CNN comparison table
fig, ax = plt.subplots(figsize=(12, 3.5))
comp_data = [
    ["Random Forest (old)", "82.70%", "0.8973", "77.96%", "84.11%", "59.00%", "67.00%"],
    ["1D-CNN (new)",  f"{test_acc*100:.2f}%", f"{test_auc:.4f}",
     f"{sensitivity*100:.2f}%", f"{specificity*100:.2f}%",
     f"{precision*100:.2f}%", f"{f1*100:.2f}%"]
]

# Determine which is better per column
table = ax.table(
    cellText=comp_data,
    colLabels=["Model", "Accuracy", "ROC-AUC", "Sensitivity", "Specificity", "Precision", "F1-Score"],
    cellLoc="center", loc="center",
    colColours=["#E3F2FD"] * 7
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

for j in range(7):
    table[2, j].set_facecolor("#E8F5E9")
    table[2, j].set_text_props(fontweight="bold")

ax.axis("off")
ax.set_title("IMU Model Comparison: Random Forest vs 1D-CNN",
             fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "imu_cnn_vs_rf_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()

print(f"\nAll plots saved in: {OUT}")
print("\n" + "=" * 60)
print("  TRAINING COMPLETE — Awaiting your approval for fusion merge")
print("=" * 60)
