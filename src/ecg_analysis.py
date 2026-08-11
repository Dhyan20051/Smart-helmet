"""
ECG Model Analysis Script
1. Overfitting Analysis: Train vs Validation accuracy/loss per epoch
2. ROC-AUC vs Sampling Frequency graph (uses existing experiment data + re-runs if needed)
"""
import os, sys, io, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import wfdb
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

SCRIPT_DIR  = os.path.dirname(__file__)
BASE_DIR    = os.path.dirname(SCRIPT_DIR)                     # D:\Mini Project
DATASET_PATH = os.path.join(BASE_DIR, "mit-bih-arrhythmia-database-p-wave-annotations")
OUT = os.path.join(SCRIPT_DIR, "ecg_analysis_plots")
os.makedirs(OUT, exist_ok=True)

TRAIN_RECORDS = ["100", "101", "103", "106", "117"]
TEST_RECORDS  = ["119", "122", "214", "223"]
ORIGINAL_FS   = 360
WINDOW_SEC    = 1.0
EPOCHS        = 30
BATCH_SIZE    = 64

# ==============================================================
# HELPERS
# ==============================================================
def is_far(idx, pwaves, margin):
    return np.all(np.abs(pwaves - idx) > margin)

def load_ecg_windows(records, fs, dataset_path):
    window = int(fs * WINDOW_SEC)
    half   = window // 2
    pw_margin = int(0.12 * fs)
    X, y = [], []

    for rec in records:
        record = wfdb.rdrecord(os.path.join(dataset_path, rec))
        ann    = wfdb.rdann(os.path.join(dataset_path, rec), "pwave")
        ecg    = record.p_signal[:, 0]
        pw     = ann.sample

        if fs != ORIGINAL_FS:
            new_len = int(len(ecg) * fs / ORIGINAL_FS)
            ecg = resample(ecg, new_len)
            pw  = (pw * fs / ORIGINAL_FS).astype(int)

        for s in pw:
            if s - half < 0 or s + half >= len(ecg): continue
            X.append(ecg[s - half : s + half])
            y.append(1)

        neg, att = 0, 0
        while neg < len(pw) and att < len(pw) * 10:
            idx = np.random.randint(half, len(ecg) - half); att += 1
            if is_far(idx, pw, pw_margin):
                X.append(ecg[idx - half : idx + half])
                y.append(0); neg += 1

    X = np.array(X); y = np.array(y)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    X = X[..., np.newaxis]
    return X, y

def build_cnn(input_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_length, 1)),
        tf.keras.layers.Conv1D(32, 7, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 5, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ==============================================================
# PART 1 — OVERFITTING ANALYSIS
# ==============================================================
print("=" * 60)
print("  PART 1: OVERFITTING ANALYSIS (Train vs Test)")
print("=" * 60)

# --- Load train data (records 100,101,103,106,117) ---
print("\nLoading TRAIN data...")
X_all, y_all = load_ecg_windows(TRAIN_RECORDS, ORIGINAL_FS, DATASET_PATH)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=42
)
print(f"Train samples: {X_train.shape[0]},  Val samples: {X_val.shape[0]}")

# --- Load test data (records 119,122,214,223) ---
print("Loading TEST data...")
X_test_all, y_test_all = load_ecg_windows(TEST_RECORDS, ORIGINAL_FS, DATASET_PATH)

_, X_test, _, y_test = train_test_split(
    X_test_all, y_test_all, test_size=0.30, stratify=y_test_all, random_state=42
)
print(f"Test samples : {X_test.shape[0]}")

# --- Train model ---
print("\nTraining CNN model (30 epochs)...")
model = build_cnn(ORIGINAL_FS)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# --- Compute per-epoch test accuracy ---
print("\nComputing per-epoch test metrics...")
train_accs = history.history["accuracy"]
val_accs   = history.history["val_accuracy"]
train_loss = history.history["loss"]
val_loss   = history.history["val_loss"]

# Final test eval
test_scores = model.predict(X_test, verbose=0).ravel()
test_preds  = (test_scores > 0.5).astype(int)
test_acc    = accuracy_score(y_test, test_preds)
fpr_t, tpr_t, _ = roc_curve(y_test, test_scores)
test_auc    = auc(fpr_t, tpr_t)

train_scores = model.predict(X_train, verbose=0).ravel()
train_preds  = (train_scores > 0.5).astype(int)
train_acc_final = accuracy_score(y_train, train_preds)

print(f"\nFinal Train Accuracy : {train_acc_final:.4f}")
print(f"Final Val Accuracy   : {val_accs[-1]:.4f}")
print(f"Final Test Accuracy  : {test_acc:.4f}")
print(f"Test ROC-AUC         : {test_auc:.4f}")

# --- PLOT 1: Accuracy over epochs ---
epochs_range = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, train_accs, "b-o", markersize=4, label="Train Accuracy")
axes[0].plot(epochs_range, val_accs,   "r-s", markersize=4, label="Validation Accuracy")
axes[0].axhline(test_acc, color="green", linestyle="--", linewidth=1.5,
                label=f"Test Accuracy = {test_acc:.4f}")
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Accuracy", fontsize=12)
axes[0].set_title("ECG Model - Overfitting Analysis (Accuracy)", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.5, 1.02)

# --- PLOT 2: Loss over epochs ---
axes[1].plot(epochs_range, train_loss, "b-o", markersize=4, label="Train Loss")
axes[1].plot(epochs_range, val_loss,   "r-s", markersize=4, label="Validation Loss")
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Loss", fontsize=12)
axes[1].set_title("ECG Model - Overfitting Analysis (Loss)", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "ecg_overfitting_analysis.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"[OK] Overfitting plot saved -> {OUT}\\ecg_overfitting_analysis.png")


# ==============================================================
# PART 2 — ROC-AUC vs SAMPLING FREQUENCY
# ==============================================================
print("\n" + "=" * 60)
print("  PART 2: ROC-AUC vs SAMPLING FREQUENCY")
print("=" * 60)

FS_LIST = [360, 250, 200, 125]
fs_results = []

for fs in FS_LIST:
    print(f"\n--- Experiment: FS = {fs} Hz ---")

    window = int(fs * WINDOW_SEC)
    X_fs, y_fs = load_ecg_windows(TRAIN_RECORDS, fs, DATASET_PATH)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_fs, y_fs, test_size=0.30, stratify=y_fs, random_state=42
    )

    m = build_cnn(window)
    m.fit(X_tr, y_tr, epochs=20, batch_size=64, validation_split=0.2, verbose=0)

    # Train metrics
    tr_sc = m.predict(X_tr, verbose=0).ravel()
    tr_acc = accuracy_score(y_tr, (tr_sc > 0.5).astype(int))

    # Test metrics
    te_sc = m.predict(X_te, verbose=0).ravel()
    te_preds = (te_sc > 0.5).astype(int)
    te_acc = accuracy_score(y_te, te_preds)
    f, t, _ = roc_curve(y_te, te_sc)
    r_auc = auc(f, t)

    fs_results.append({
        "FS_Hz": fs,
        "Train_Acc": tr_acc,
        "Test_Acc": te_acc,
        "ROC_AUC": r_auc
    })
    print(f"  Train Acc: {tr_acc:.4f}  |  Test Acc: {te_acc:.4f}  |  ROC-AUC: {r_auc:.4f}")

df = pd.DataFrame(fs_results)
df.to_csv(os.path.join(OUT, "ecg_fs_experiment_results.csv"), index=False)
print("\n--- Results Table ---")
print(df.to_string(index=False))

# --- PLOT 3: ROC-AUC vs Sampling Frequency ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: ROC-AUC line + bar
axes[0].plot(df["FS_Hz"], df["ROC_AUC"], "ro-", markersize=8, linewidth=2, label="ROC-AUC")
axes[0].fill_between(df["FS_Hz"], df["ROC_AUC"], alpha=0.15, color="red")
for _, row in df.iterrows():
    axes[0].annotate(f'{row["ROC_AUC"]:.4f}',
                     (row["FS_Hz"], row["ROC_AUC"]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, fontweight="bold")
axes[0].set_xlabel("Sampling Frequency (Hz)", fontsize=12)
axes[0].set_ylabel("ROC-AUC", fontsize=12)
axes[0].set_title("ROC-AUC vs Sampling Frequency", fontsize=13, fontweight="bold")
axes[0].set_xticks(df["FS_Hz"])
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# Right: Train vs Test accuracy
x = np.arange(len(df))
w = 0.35
axes[1].bar(x - w/2, df["Train_Acc"], w, label="Train Accuracy", color="#4C72B0", edgecolor="black")
axes[1].bar(x + w/2, df["Test_Acc"],  w, label="Test Accuracy",  color="#55A868", edgecolor="black")
axes[1].set_xlabel("Sampling Frequency (Hz)", fontsize=12)
axes[1].set_ylabel("Accuracy", fontsize=12)
axes[1].set_title("Train vs Test Accuracy at Different FS", fontsize=13, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(df["FS_Hz"].astype(str))
axes[1].set_ylim(0.95, 1.005)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis="y")

for i, (tr, te) in enumerate(zip(df["Train_Acc"], df["Test_Acc"])):
    axes[1].text(i - w/2, tr + 0.002, f"{tr:.3f}", ha="center", fontsize=8)
    axes[1].text(i + w/2, te + 0.002, f"{te:.3f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "ecg_roc_vs_frequency.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"\n[OK] ROC-AUC vs FS plot saved -> {OUT}\\ecg_roc_vs_frequency.png")

print("\n" + "=" * 60)
print("  ALL ECG ANALYSIS COMPLETE")
print("=" * 60)
print(f"All plots saved in: {OUT}")
