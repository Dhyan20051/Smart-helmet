import os
import time
import numpy as np
import tensorflow as tf
import joblib

# ======================================================
# PATHS — all models and samples in final_programs/
# ======================================================
BASE_DIR = os.path.dirname(__file__)

ECG_MODEL_PATH = os.path.join(BASE_DIR, "ecg_model_70.h5")
EEG_MODEL_PATH = os.path.join(BASE_DIR, "eeg_xgboost_full_14.pkl")
IMU_MODEL_PATH = os.path.join(BASE_DIR, "imu_mobifall_crash_model_new.pkl")

ECG_SAMPLE_PATH = os.path.join(BASE_DIR, "sample_ecg.npy")
EEG_SAMPLE_PATH = os.path.join(BASE_DIR, "sample_eeg.npy")
IMU_SAMPLE_PATH = os.path.join(BASE_DIR, "sample_imu.npy")

# ======================================================
# LOAD MODELS
# ======================================================
print("\nLoading models...")

ecg_model = tf.keras.models.load_model(ECG_MODEL_PATH)
eeg_model = joblib.load(EEG_MODEL_PATH)
imu_model = joblib.load(IMU_MODEL_PATH)

print("All models loaded successfully")

# ======================================================
# LOAD SENSOR SAMPLES
# ======================================================
sample_ecg = np.load(ECG_SAMPLE_PATH).reshape(1, 360, 1)
sample_eeg = np.load(EEG_SAMPLE_PATH).reshape(1, -1)
sample_imu_base = np.load(IMU_SAMPLE_PATH)

# ======================================================
# IMU FEATURE EXTRACTION (12 FEATURES - MUST MATCH TRAINING)
# ======================================================
def extract_imu_features(window):
    """
    window shape: (N, 3) -> ax, ay, az
    returns: (1, 12)
    """
    ax = window[:, 0]
    ay = window[:, 1]
    az = window[:, 2]

    features = [
        np.mean(ax), np.std(ax), np.max(ax), np.min(ax),
        np.mean(ay), np.std(ay), np.max(ay), np.min(ay),
        np.mean(az), np.std(az), np.max(az), np.min(az)
    ]

    return np.array(features).reshape(1, -1)

# ======================================================
# FUSION WEIGHTS
# ======================================================
W_IMU = 0.5
W_ECG = 0.3
W_EEG = 0.2

# ======================================================
# STREAMING SIMULATION
# ======================================================

print("\nStarting DEMO fusion simulation...\n")

t = 0
while True:
    t += 1
    print(f"\nTime step: {t}")

    # ==================================================
    # SCENARIO CONTROL (DEMO LOGIC)
    # ==================================================
    if t <= 10:
        scenario = "NORMAL"
    elif t <= 20:
        scenario = "FATIGUE"
    elif t <= 30:
        scenario = "CARDIAC"
    else:
        scenario = "CRASH"

    # ================= ECG =================
    if scenario == "CARDIAC":
        ecg_prob = np.random.uniform(0.88, 0.96)   # Abnormal ECG
    else:
        ecg_prob = np.random.uniform(0.20, 0.50)   # Normal ECG

    # ================= EEG =================
    if scenario == "FATIGUE":
        eeg_prob = np.random.uniform(0.75, 0.90)
    else:
        eeg_prob = np.random.uniform(0.20, 0.50)

    eeg_alert = eeg_prob > 0.7

    # ================= IMU =================
    imu_window = sample_imu_base.copy()

    if scenario == "CRASH":
        imu_window += np.random.normal(7, 3, imu_window.shape)
    else:
        imu_window += np.random.normal(0, 0.05, imu_window.shape)

    imu_features = extract_imu_features(imu_window)
    imu_prob = imu_model.predict_proba(imu_features)[0][1]
    imu_pred = int(imu_prob > 0.5)

    # ================= FUSION =================
    fusion_risk = (
        0.5 * imu_prob +
        0.3 * ecg_prob +
        0.2 * eeg_prob
    )

    # ================= OUTPUT =================
    print(f"Scenario               : {scenario}")
    print(f"ECG probability        : {ecg_prob:.3f}")
    print(f"EEG fatigue probability: {eeg_prob:.3f}")
    print(f"IMU crash probability  : {imu_prob:.3f}")
    print(f"Fusion risk score      : {fusion_risk:.3f}")

    # ================= ALERT LOGIC =================
    if scenario == "CRASH":
        print("CRASH ALERT - Emergency services triggered")
    elif scenario == "CARDIAC":
        print("CARDIAC ALERT - Abnormal heart activity detected")
    elif scenario == "FATIGUE":
        print("FATIGUE ALERT - Driver drowsy")
    else:
        print("Driver state normal")

    time.sleep(1)
