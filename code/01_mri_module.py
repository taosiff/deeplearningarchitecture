"""
01_mri_module.py
================
Module 1: Structural MRI Pipeline (Pretrained 3D-ResNet-18 + Strict Train-Only PCA + Balanced Metric Head)
prev_archi_pro_max (Zero-Leakage & Transparent Q1 Revision)

Methodological Compliance:
1. Strict Training Isolation:
   - StandardScaler and PCA (k=5) are fit strictly on data/splits/train.csv (n=180, 24 pos, 90 families).
   - Validation (n=62, 7 pos, 31 families) and Test (n=62, 7 pos, 31 families) are strictly transformed.
   - Zero validation or test data enters PCA or classifier fitting.
2. Balanced Linear Metric Head:
   - LogisticRegression(C=0.1, class_weight='balanced') fit strictly on train.csv.
   - Evaluated on val.csv and held-out test.csv with zero threshold/hyperparameter peeking.
3. Saves:
   - data/processed/m1_train_embeddings.npy, m1_val_embeddings.npy, m1_test_embeddings.npy
   - data/processed/m1_train_probs.npy, m1_val_probs.npy, m1_test_probs.npy
   - results/test_probs_m1.npy
   - models/m1_mri_model.pkl, models/m1_mri_metadata.json
"""

import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
)

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
SPLITS_DIR = DATA_DIR / 'splits'
MRI_DIR = DATA_DIR / 'mri'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
OUT_DIR = BASE_DIR / 'results'

for d in [PROCESSED_DIR, MODELS_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MODULE 1: sMRI PRETRAINED RESNET-18 (STRICT TRAIN-ONLY PCA & METRIC HEAD)")
print("=" * 80)

# 1. Load splits
train_split = pd.read_csv(SPLITS_DIR / 'train.csv')
val_split = pd.read_csv(SPLITS_DIR / 'val.csv')
test_split = pd.read_csv(SPLITS_DIR / 'test.csv')

train_ids = train_split['participant_id'].astype(str).values
val_ids = val_split['participant_id'].astype(str).values
test_ids = test_split['participant_id'].astype(str).values

train_fams = set(train_split['family_id'].astype(str).values)
val_fams = set(val_split['family_id'].astype(str).values)
test_fams = set(test_split['family_id'].astype(str).values)

# Verify complete family isolation across all three partitions
assert len(train_fams.intersection(val_fams)) == 0, "CRITICAL: Family contamination between train and val!"
assert len(train_fams.intersection(test_fams)) == 0, "CRITICAL: Family contamination between train and test!"
assert len(val_fams.intersection(test_fams)) == 0, "CRITICAL: Family contamination between val and test!"

y_train = train_split['internalizing_incident'].values.astype(int)
y_val = val_split['internalizing_incident'].values.astype(int)
y_test = test_split['internalizing_incident'].values.astype(int)

print(f"Cohort Partitions: Train n = {len(train_ids)} ({sum(y_train)} pos), "
      f"Val n = {len(val_ids)} ({sum(y_val)} pos), "
      f"Test n = {len(test_ids)} ({sum(y_test)} pos)")

# 2. Load 768-D pretrained Med3D embeddings (extracted from sMRI T1w scans)
mri_data = np.load(MRI_DIR / 'standalone__pretrained.npz', allow_pickle=True)
mri_feats = mri_data['features']
mri_pids = [str(p) for p in mri_data['pids']]
pid_to_idx = {pid: i for i, pid in enumerate(mri_pids)}

X_train_raw = np.array([mri_feats[pid_to_idx[p]] for p in train_ids], dtype=float)
X_val_raw = np.array([mri_feats[pid_to_idx[p]] for p in val_ids], dtype=float)
X_test_raw = np.array([mri_feats[pid_to_idx[p]] for p in test_ids], dtype=float)

# 3. Fit Scaler and PCA strictly on Train partition (Zero Leakage)
scaler_m = StandardScaler()
X_train_scaled = scaler_m.fit_transform(X_train_raw)
X_val_scaled = scaler_m.transform(X_val_raw)
X_test_scaled = scaler_m.transform(X_test_raw)

pca_m = PCA(n_components=5, random_state=SEED)
Z_mri_train_5d = pca_m.fit_transform(X_train_scaled)
Z_mri_val_5d = pca_m.transform(X_val_scaled)
Z_mri_test_5d = pca_m.transform(X_test_scaled)

var_exp = np.sum(pca_m.explained_variance_ratio_) * 100
print(f"sMRI PCA-5 Explained Variance Ratio (Train fit): {var_exp:.2f}%")

# 4. Fit Balanced Metric Head strictly on Train
clf_m = LogisticRegression(C=0.1, class_weight='balanced', random_state=SEED, max_iter=1000)
clf_m.fit(Z_mri_train_5d, y_train)

p_train_m1 = clf_m.predict_proba(Z_mri_train_5d)[:, 1]
p_val_m1 = clf_m.predict_proba(Z_mri_val_5d)[:, 1]
p_test_m1 = clf_m.predict_proba(Z_mri_test_5d)[:, 1]

val_auc_m1 = roc_auc_score(y_val, p_val_m1)
val_pr_m1 = average_precision_score(y_val, p_val_m1)
test_auc_m1 = roc_auc_score(y_test, p_test_m1)
test_pr_m1 = average_precision_score(y_test, p_test_m1)

# Dynamic metrics on test (threshold = 0.5)
cm = confusion_matrix(y_test, (p_test_m1 >= 0.5).astype(int), labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
brier = brier_score_loss(y_test, p_test_m1)

print(f"\nModule 1 Validation Evaluation:")
print(f"  Val ROC-AUC:      {val_auc_m1:.4f}")
print(f"  Val PR-AUC:       {val_pr_m1:.4f}")

print(f"\nModule 1 Held-Out Test Evaluation:")
print(f"  Test ROC-AUC:     {test_auc_m1:.4f}")
print(f"  Test PR-AUC:      {test_pr_m1:.4f}")
print(f"  Test Sensitivity: {sens*100:.2f}% ({tp}/{tp+fn})")
print(f"  Test Specificity: {spec*100:.2f}% ({tn}/{tn+fp})")
print(f"  Test Brier:       {brier:.4f}")

# 5. Save processed embeddings and predictions
np.save(PROCESSED_DIR / 'm1_train_embeddings.npy', Z_mri_train_5d)
np.save(PROCESSED_DIR / 'm1_val_embeddings.npy', Z_mri_val_5d)
np.save(PROCESSED_DIR / 'm1_test_embeddings.npy', Z_mri_test_5d)

np.save(PROCESSED_DIR / 'm1_train_probs.npy', p_train_m1)
np.save(PROCESSED_DIR / 'm1_val_probs.npy', p_val_m1)
np.save(PROCESSED_DIR / 'm1_test_probs.npy', p_test_m1)
np.save(OUT_DIR / 'test_probs_m1.npy', p_test_m1)

# Combined pool representation for compatibility if needed
Z_mri_pool_5d = np.vstack([Z_mri_train_5d, Z_mri_val_5d])
np.save(PROCESSED_DIR / 'Z_mri_pool_5d.npy', Z_mri_pool_5d)
np.save(PROCESSED_DIR / 'Z_mri_test_5d.npy', Z_mri_test_5d)

with open(MODELS_DIR / 'm1_mri_model.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler_m, 'pca': pca_m, 'clf': clf_m}, f)

meta = {
    'module': 'Module 1 (Structural Neuroimaging / sMRI)',
    'backbone': 'Pretrained 3D ResNet-18 (Med3D)',
    'dims': 5,
    'training_scope': 'Strictly train.csv (n=180, zero val/test leakage)',
    'val_roc_auc': round(val_auc_m1, 4),
    'val_pr_auc': round(val_pr_m1, 4),
    'test_roc_auc': round(test_auc_m1, 4),
    'test_pr_auc': round(test_pr_m1, 4),
    'test_sens': round(sens, 4),
    'test_spec': round(spec, 4),
    'test_brier': round(brier, 4)
}
(MODELS_DIR / 'm1_mri_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print("\nModule 1 execution complete.")
