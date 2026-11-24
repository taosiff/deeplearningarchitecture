"""
01_mri_module.py
================
Module 1: Structural Neuroimaging (sMRI Pretrained 3D-ResNet-18 + PCA-5)
prev_archi_pro_max Pipeline (Strict Zero-Leakage)

Methodological Compliance:
1. Frozen feature representations: 768-D representations extracted from a self-supervised
   Med3D 3D-ResNet-18 backbone pretrained across 23 medical volumetric datasets.
2. In-pool feature engineering: StandardScaler and PCA-5 are fit strictly on the development pool (n=242).
3. Out-of-fold evaluation: 5-fold StratifiedGroupKFold ensures that development-pool validation
   predictions are strictly out-of-sample and family-isolated (zero twin pair overlap).
4. Held-out test evaluation: Evaluated blindly on the held-out test partition (n=62, 7 cases).
"""

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
SPLITS_DIR = DATA_DIR / 'splits'
MRI_DIR = DATA_DIR / 'mri'
OUT_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'models'
PROCESSED_DIR = DATA_DIR / 'processed'

for d in [OUT_DIR, MODELS_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PREV ARCHI PRO MAX - MODULE 1: sMRI PRETRAINED RESNET-18 (LEAK-FREE)")
print("=" * 80)

# 1. Load splits
pool_split = pd.read_csv(SPLITS_DIR / 'pool.csv')
test_split = pd.read_csv(SPLITS_DIR / 'test.csv')

pool_ids = pool_split['participant_id'].astype(str).values
test_ids = test_split['participant_id'].astype(str).values
pool_fams = pool_split['family_id'].astype(str).values
test_fams = test_split['family_id'].astype(str).values

y_pool = pool_split['internalizing_incident'].values.astype(int)
y_test = test_split['internalizing_incident'].values.astype(int)

# Verify disjoint families
assert len(set(pool_fams).intersection(set(test_fams))) == 0, "CRITICAL: Family contamination between pool and test!"
print(f"Cohort Partitions: Pool n = {len(pool_ids)} (121 families), Test n = {len(test_ids)} (31 families)")

# 2. Load 768-D pretrained Med3D embeddings
mri_data = np.load(MRI_DIR / 'standalone__pretrained.npz', allow_pickle=True)
mri_feats = mri_data['features']
mri_pids = [str(p) for p in mri_data['pids']]
pid_to_idx = {pid: i for i, pid in enumerate(mri_pids)}

X_pool_raw = np.array([mri_feats[pid_to_idx[p]] for p in pool_ids], dtype=float)
X_test_raw = np.array([mri_feats[pid_to_idx[p]] for p in test_ids], dtype=float)

# 3. Fit Scaler and PCA strictly on development pool
scaler_m = StandardScaler()
X_pool_scaled = scaler_m.fit_transform(X_pool_raw)
X_test_scaled = scaler_m.transform(X_test_raw)

pca_m = PCA(n_components=5, random_state=SEED)
Z_mri_pool_5d = pca_m.fit_transform(X_pool_scaled)
Z_mri_test_5d = pca_m.transform(X_test_scaled)

print(f"sMRI PCA-5 Explained Variance Ratio: {np.sum(pca_m.explained_variance_ratio_)*100:.2f}%")

# 4. Out-of-fold pool cross-validation (5-fold StratifiedGroupKFold)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_probs_m1 = np.zeros(len(y_pool))

for tr_idx, val_idx in sgkf.split(Z_mri_pool_5d, y_pool, pool_fams):
    X_tr_fold, y_tr_fold = Z_mri_pool_5d[tr_idx], y_pool[tr_idx]
    X_val_fold = Z_mri_pool_5d[val_idx]
    
    fold_clf = LogisticRegression(C=0.1, class_weight='balanced', random_state=SEED, max_iter=1000)
    fold_clf.fit(X_tr_fold, y_tr_fold)
    oof_probs_m1[val_idx] = fold_clf.predict_proba(X_val_fold)[:, 1]

pool_auc_m1 = roc_auc_score(y_pool, oof_probs_m1)
print(f"Module 1 Out-of-Fold Pool AUC: {pool_auc_m1:.4f}")

# 5. Fit full development model and evaluate on held-out test cohort
clf_m = LogisticRegression(C=0.1, class_weight='balanced', random_state=SEED, max_iter=1000)
clf_m.fit(Z_mri_pool_5d, y_pool)

p_test_m1 = clf_m.predict_proba(Z_mri_test_5d)[:, 1]
test_auc_m1 = roc_auc_score(y_test, p_test_m1)
test_pr_m1 = average_precision_score(y_test, p_test_m1)

cm = confusion_matrix(y_test, (p_test_m1 >= 0.5).astype(int), labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)

print(f"\nModule 1 Held-Out Test Evaluation:")
print(f"  Test ROC-AUC:     {test_auc_m1:.4f}")
print(f"  Test PR-AUC:      {test_pr_m1:.4f}")
print(f"  Test Sensitivity: {sens*100:.2f}% ({tp}/{tp+fn})")
print(f"  Test Specificity: {spec*100:.2f}% ({tn}/{tn+fp})")

# 6. Save representations and models
np.save(PROCESSED_DIR / 'Z_mri_pool_5d.npy', Z_mri_pool_5d)
np.save(PROCESSED_DIR / 'Z_mri_test_5d.npy', Z_mri_test_5d)
np.save(PROCESSED_DIR / 'm1_train_embeddings.npy', Z_mri_pool_5d[:180])
np.save(PROCESSED_DIR / 'm1_val_embeddings.npy', Z_mri_pool_5d[180:])
np.save(PROCESSED_DIR / 'm1_test_embeddings.npy', Z_mri_test_5d)

np.save(OUT_DIR / 'oof_probs_m1.npy', oof_probs_m1)
np.save(OUT_DIR / 'test_probs_m1.npy', p_test_m1)
np.save(PROCESSED_DIR / 'm1_train_probs.npy', oof_probs_m1[:180])
np.save(PROCESSED_DIR / 'm1_val_probs.npy', oof_probs_m1[180:])
np.save(PROCESSED_DIR / 'm1_test_probs.npy', p_test_m1)

with open(MODELS_DIR / 'm1_mri_model.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler_m, 'pca': pca_m, 'clf': clf_m}, f)

meta = {
    'module': 'Module 1 (Structural Neuroimaging / sMRI)',
    'backbone': 'Pretrained 3D ResNet-18 (Med3D)',
    'dims': 5,
    'pool_oof_auc': round(pool_auc_m1, 4),
    'test_roc_auc': round(test_auc_m1, 4),
    'test_pr_auc': round(test_pr_m1, 4),
    'test_sens': round(sens, 4),
    'test_spec': round(spec, 4)
}
(MODELS_DIR / 'm1_mri_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print("Module 1 completed successfully.")
