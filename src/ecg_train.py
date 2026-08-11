import os
import numpy as np
import wfdb
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ======================================================
# CONFIGURATION
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # → D:\Mini Project
DATASET_PATH = os.path.join(BASE_DIR, "mit-bih-arrhythmia-database-p-wave-annotations")
RECORDS = ["100", "101", "103", "106", "117"]

FS = 360                    # Sampling frequency (Hz)
WINDOW = FS                 # 1-second window
P_WAVE_WIDTH = int(0.12*FS) # ~120 ms physiological P-wave
EPOCHS = 30
BATCH_SIZE = 64
RANDOM_STATE = 42

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "ecg_model_70.h5")

# ======================================================
# HELPER FUNCTION
# ======================================================
def is_far_from_pwaves(idx, pwave_samples, margin):
    """True if idx is outside P-wave region"""
    return np.all(np.abs(pwave_samples - idx) > margin)

# ======================================================
# LOAD DATA + BALANCED WINDOW CREATION
# ======================================================
X, y = [], []

print("\nLoading ECG data with physiologically correct sampling...\n")

for rec in RECORDS:
    print(f"Processing record: {rec}")

    record = wfdb.rdrecord(os.path.join(DATASET_PATH, rec))
    ann = wfdb.rdann(os.path.join(DATASET_PATH, rec), "pwave")

    ecg = record.p_signal[:, 0]
    pwave_samples = ann.sample

    # -----------------------------
    # POSITIVE CLASS (P-wave)
    # -----------------------------
    for s in pwave_samples:
        if s - WINDOW//2 < 0 or s + WINDOW//2 >= len(ecg):
            continue
        X.append(ecg[s - WINDOW//2 : s + WINDOW//2])
        y.append(1)

    # -----------------------------
    # NEGATIVE CLASS (Non-P-wave)
    # -----------------------------
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

    if neg_count < required_neg:
        print(f"Warning: only {neg_count}/{required_neg} negatives for record {rec}")

# ======================================================
# CONVERT + NORMALIZE
# ======================================================
X = np.array(X)
y = np.array(y)

print("\nDataset summary:")
print("Total samples:", X.shape[0])
print("Label distribution:", np.unique(y, return_counts=True))

# Per-window normalization
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
X = X[..., np.newaxis]

# ======================================================
# TRAIN SPLIT — 70% ONLY
# ======================================================
X_train, _, y_train, _ = train_test_split(
    X,
    y,
    train_size=0.70,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTraining samples (70%):", X_train.shape)

# ======================================================
# CNN MODEL
# ======================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW, 1)),

    tf.keras.layers.Conv1D(32, 7, activation="relu"),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(64, 5, activation="relu"),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    ),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================================================
# TRAINING
# ======================================================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ======================================================
# TRAINING METRICS
# ======================================================
train_scores = model.predict(X_train).ravel()
train_preds = (train_scores > 0.5).astype(int)

train_acc = accuracy_score(y_train, train_preds)
train_cm = confusion_matrix(y_train, train_preds)
fpr, tpr, _ = roc_curve(y_train, train_scores)
train_auc = auc(fpr, tpr)

print("\n================ TRAINING RESULTS (70%) ================")
print("Training Accuracy:", train_acc)
print("Confusion Matrix:\n", train_cm)
print("ROC-AUC:", train_auc)

# ======================================================
# SAVE MODEL
# ======================================================
model.save(MODEL_SAVE_PATH)
print("\nModel saved as:", MODEL_SAVE_PATH)

# ======================================================
# ROC CURVE
# ======================================================
plt.figure()
plt.plot(fpr, tpr, label=f"Train ROC (AUC = {train_auc:.3f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Training ROC Curve - ECG P-wave Detection")
plt.legend()
plt.show()

print("\nFinal Training Accuracy:", history.history["accuracy"][-1])
print("Final Validation Accuracy:", history.history["val_accuracy"][-1])
