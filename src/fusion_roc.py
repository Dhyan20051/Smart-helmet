import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score

OUT = r"d:\Mini Project\final_programs\fusion_dashboard"
os.makedirs(OUT, exist_ok=True)

# ==============================================================
# GENERATE LARGE-SCALE SIMULATED DATA (10,000 samples)
# Since we don't have a dataset with all 3 sensors recorded
# simultaneously, we use the known probability distributions
# of our individual models to simulate the fusion performance.
# ==============================================================
np.random.seed(42)
n_samples = 2500

# 1. NORMAL SCENARIO
ecg_norm = np.random.uniform(0.10, 0.50, n_samples)
eeg_norm = np.random.uniform(0.10, 0.50, n_samples)
imu_norm = np.random.uniform(0.01, 0.30, n_samples)

# 2. FATIGUE SCENARIO
ecg_fat = np.random.uniform(0.10, 0.50, n_samples)
eeg_fat = np.random.uniform(0.70, 0.95, n_samples)
imu_fat = np.random.uniform(0.01, 0.30, n_samples)

# 3. CARDIAC SCENARIO
ecg_car = np.random.uniform(0.85, 0.98, n_samples)
eeg_car = np.random.uniform(0.10, 0.50, n_samples)
imu_car = np.random.uniform(0.01, 0.30, n_samples)

# 4. CRASH SCENARIO
ecg_cra = np.random.uniform(0.60, 0.95, n_samples)
eeg_cra = np.random.uniform(0.60, 0.95, n_samples)
imu_cra = np.random.uniform(0.75, 0.99, n_samples)

# Combine all
ecg_all = np.concatenate([ecg_norm, ecg_fat, ecg_car, ecg_cra])
eeg_all = np.concatenate([eeg_norm, eeg_fat, eeg_car, eeg_cra])
imu_all = np.concatenate([imu_norm, imu_fat, imu_car, imu_cra])

# True labels: 0 for Normal, 1 for Any Abnormal State (Fatigue, Cardiac, Crash)
y_true = np.concatenate([np.zeros(n_samples), np.ones(n_samples * 3)])

# ==============================================================
# FUSION LOGIC (MAX-POOLING APPROACH FOR OVERALL ABNORMALITY)
# To create a single ROC curve, we need a single continuous 
# "Abnormality Score". We use the maximum probability across 
# the weighted sensors to represent the overall system alarm level.
# ==============================================================
# Applying thresholds from the simulation logic
# We scale the probabilities so they all cross a 0.5 threshold for an alert
abnormality_score = np.maximum.reduce([
    imu_all,                     # IMU threshold is 0.5
    ecg_all * (0.5 / 0.85),      # ECG threshold is 0.85 -> scale to 0.5
    eeg_all * (0.5 / 0.70)       # EEG threshold is 0.70 -> scale to 0.5
])

# Predictions: if abnormality_score > 0.5, it's an alert
y_pred = (abnormality_score > 0.5).astype(int)

# Calculate Metrics
accuracy = accuracy_score(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, abnormality_score)
roc_auc = auc(fpr, tpr)

print(f"Fusion System Theoretical Accuracy : {accuracy * 100:.2f}%")
print(f"Fusion System Theoretical ROC-AUC  : {roc_auc:.4f}")

# ==============================================================
# PLOT ROC CURVE
# ==============================================================
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#E91E63', lw=3, label=f'Fusion System ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Fusion System - Theoretical ROC Curve\n(Normal vs Abnormal State)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

# Add text box with accuracy
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
plt.text(0.6, 0.2, f"Accuracy: {accuracy*100:.2f}%\nSimulated Samples: 10,000", 
         fontsize=11, bbox=props)

save_path = os.path.join(OUT, "06_fusion_theoretical_roc.png")
plt.tight_layout()
plt.savefig(save_path, dpi=200)
plt.close()

print(f"\nPlot saved to: {save_path}")
