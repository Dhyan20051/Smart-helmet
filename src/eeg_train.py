"""
EEG Eye State Classification - Training Script
Trains XGBoost model on 70% training set with 5-fold CV validation
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
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

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = os.path.dirname(__file__)  # final_programs
JEEVITHA_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "jeevitha model")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "eeg_xgboost_full_14.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "eval_results")
os.makedirs(RESULTS_PATH, exist_ok=True)

TRAIN_FILE = os.path.join(JEEVITHA_MODEL_PATH, "full_train_14.csv")
VAL_FILE = os.path.join(JEEVITHA_MODEL_PATH, "full_val_14.csv")
TEST_FILE = os.path.join(JEEVITHA_MODEL_PATH, "full_test_14.csv")

# ==========================================================
# LOAD DATA
# ==========================================================
def load_split(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

X_train, y_train = load_split(TRAIN_FILE)
X_val, y_val = load_split(VAL_FILE)
X_test, y_test = load_split(TEST_FILE)

print("✅ EEG datasets loaded")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# ==========================================================
# OPTIMIZED XGBOOST MODEL
# ==========================================================
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=1,
    gamma=0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# ==========================================================
# CROSS-VALIDATION
# ==========================================================
print("\n📊 Running 5-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"CV ROC-AUC Scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==========================================================
# TRAINING
# ==========================================================
print("\n🚀 Training EEG XGBoost model...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# ==========================================================
# VALIDATION
# ==========================================================
val_pred = model.predict(X_val)
val_prob = model.predict_proba(X_val)[:, 1]
val_acc = accuracy_score(y_val, val_pred)
val_auc = roc_auc_score(y_val, val_prob)
print(f"🧪 Validation Accuracy: {val_acc:.4f}")
print(f"🧪 Validation ROC-AUC: {val_auc:.4f}")

# ==========================================================
# TESTING
# ==========================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print("\n" + "="*50)
print("  EEG TEST RESULTS (OPTIMIZED MODEL)")
print("="*50)
print(f"✅ Test Accuracy   : {test_acc:.4f}")
print(f"📈 ROC–AUC         : {roc_auc:.4f}")
print(f"📊 F1-Score        : {f1:.4f}")
print(f"🎯 Precision       : {precision:.4f}")
print(f"🔍 Recall          : {recall:.4f}")
print(f"📌 Sensitivity     : {sensitivity:.4f}")
print(f"📌 Specificity     : {specificity:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Eyes Open", "Eyes Closed"]))

print("Confusion Matrix:")
print(cm)

# ==========================================================
# SAVE MODEL & RESULTS
# ==========================================================
joblib.dump(model, MODEL_SAVE_PATH)
print(f"\n💾 Model saved at: {MODEL_SAVE_PATH}")

results_df = pd.DataFrame({
    'Metric': ['Test Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Sensitivity', 'Specificity'],
    'Value': [test_acc, roc_auc, f1, precision, recall, sensitivity, specificity]
})
results_df.to_csv(os.path.join(RESULTS_PATH, 'eeg_train_results.csv'), index=False)
print(f"📋 Results saved -> {RESULTS_PATH}\\eeg_train_results.csv")
