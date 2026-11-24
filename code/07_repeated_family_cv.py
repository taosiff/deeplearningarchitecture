"""
07_repeated_family_cv.py
========================
Repeated Family-Stratified Cross-Validation Pipeline (M6 / Recommendation 1+4)
prev_archi_pro_max (Strict Zero-Leakage & Transparent Q1 Revision)

Methodological Compliance:
1. Exact Cohort Alignment:
   - Aggregates the exact 304 participants, 152 twin families, and 38 incident cases from data/splits/
     (train.csv + val.csv + test.csv), eliminating the 7-case label discrepancy from raw outcomes.csv.
2. Family-Isolated 5x5 Folds (25 Total):
   - 5 independent random repeats of 5-fold StratifiedGroupKFold on family_id.
   - Zero twin pair separation: both twins of any family always fall together in train or test fold.
3. Strict Train-Only Transformation & Zero In-Sample Thresholding:
   - Scaler and PCA (k=5 for sMRI) fit strictly on the 4/5 training folds in every iteration.
   - Optimal decision threshold (Youden's J) derived exclusively on the training fold, then frozen
     and evaluated blindly on the out-of-fold validation fold.
4. Statistically Sound Variance Reporting:
   - Reports Mean, SD, Median, Interquartile Range (IQR [25th, 75th]), and Range [Min, Max] across
     the 25 folds, explicitly acknowledging that overlapping CV folds cannot be treated as independent i.i.d. draws.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_curve,
    confusion_matrix,
    brier_score_loss
)

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
SPLITS_DIR = DATA_DIR / 'splits'
RAW_DIR = DATA_DIR / 'raw'
MRI_DIR = DATA_DIR / 'mri'
OUT_DIR = BASE_DIR / 'results'

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("07_REPEATED_FAMILY_CV: 5x5 REPEATED FAMILY-STRATIFIED CROSS-VALIDATION")
print("=" * 80)

# 1. Load exact 304 participants from split files (38 true incident cases, 152 families)
train_split = pd.read_csv(SPLITS_DIR / 'train.csv')
val_split = pd.read_csv(SPLITS_DIR / 'val.csv')
test_split = pd.read_csv(SPLITS_DIR / 'test.csv')

splits_all = pd.concat([train_split, val_split, test_split], ignore_index=True)
all_pids = splits_all['participant_id'].astype(str).values
all_fams = splits_all['family_id'].astype(str).values
all_y = splits_all['internalizing_incident'].values.astype(int)

n_total = len(all_pids)
n_pos = sum(all_y)
n_fams = len(np.unique(all_fams))

print(f"Cohort Definition (from splits): N={n_total} participants, {n_pos} incident cases ({n_pos/n_total*100:.2f}%), "
      f"{n_fams} unique twin families (zero split-outcome discrepancy)")

# 2. Load Modality Features
# Module 1: Pretrained Med3D sMRI Embeddings
mri_data = np.load(MRI_DIR / 'standalone__pretrained.npz', allow_pickle=True)
mri_feats_raw = mri_data['features']
mri_pids_list = [str(p) for p in mri_data['pids']]
mri_pid_map = {pid: i for i, pid in enumerate(mri_pids_list)}
X_m1_all = np.array([mri_feats_raw[mri_pid_map[p]] for p in all_pids], dtype=float)

# Module 2: Questionnaire Transition Features (9 items)
quest_df = pd.read_csv(RAW_DIR / 'questionnaire.csv')
m2_merged = pd.DataFrame({'participant_id': all_pids}).merge(quest_df, on='participant_id', how='left')
diff_26 = pd.to_numeric(m2_merged['pSCAS26'], errors='coerce') - pd.to_numeric(m2_merged['SCAS26'], errors='coerce')
diff_worry = pd.to_numeric(m2_merged['pSCAS31'], errors='coerce') - pd.to_numeric(m2_merged['SCAS31'], errors='coerce')
X_m2_all = pd.DataFrame({
    'pSDQ03': pd.to_numeric(m2_merged['pSDQ03'], errors='coerce'),
    'SPHERE07': pd.to_numeric(m2_merged['SPHERE07'], errors='coerce'),
    'SPHERE08': pd.to_numeric(m2_merged['SPHERE08'], errors='coerce'),
    'SPHERE22': pd.to_numeric(m2_merged['SPHERE22'], errors='coerce'),
    'pSCAS31': pd.to_numeric(m2_merged['pSCAS31'], errors='coerce'),
    'pSCAS_socpho_score': pd.to_numeric(m2_merged['pSCAS_socpho_score'], errors='coerce'),
    'SCAS_socpho_score': pd.to_numeric(m2_merged['SCAS_socpho_score'], errors='coerce'),
    'diff_26': diff_26,
    'diff_worry': diff_worry
}).values

# Module 3: Phenotypic Transition Features (4 items)
pheno_df = pd.read_csv(RAW_DIR / 'phenotypic.csv')
m3_merged = pd.DataFrame({'participant_id': all_pids}).merge(pheno_df, on='participant_id', how='left')
p_cols = ['birth_IER_decile', 'birth_IRSD_decile', 'birth_IRSAD_decile', 'pPDS01_ses01']
X_m3_all = m3_merged[p_cols].apply(pd.to_numeric, errors='coerce').values

# 3. 5x5 Repeated Family-Stratified Cross-Validation Execution
n_repeats = 5
n_splits = 5
total_folds = n_repeats * n_splits

fold_results = {
    'm1': {'auc': [], 'pr': [], 'f1': [], 'sens': [], 'spec': []},
    'm2': {'auc': [], 'pr': [], 'f1': [], 'sens': [], 'spec': []},
    'm3': {'auc': [], 'pr': [], 'f1': [], 'sens': [], 'spec': []},
    'fusion_avg': {'auc': [], 'pr': [], 'f1': [], 'sens': [], 'spec': []}
}

print(f"\nExecuting {n_repeats} repeats of {n_splits}-fold StratifiedGroupKFold ({total_folds} total family-isolated folds)...")

fold_counter = 0
for rep in range(n_repeats):
    rep_seed = SEED + rep * 101
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep_seed)
    
    for tr_idx, val_idx in sgkf.split(all_pids, all_y, all_fams):
        fold_counter += 1
        
        # Verify complete family isolation in this fold
        tr_fams = set(all_fams[tr_idx])
        val_fams = set(all_fams[val_idx])
        assert len(tr_fams.intersection(val_fams)) == 0, f"Contamination in fold {fold_counter}!"
        
        y_tr, y_val = all_y[tr_idx], all_y[val_idx]
        
        # -------------------------------------------------------------
        # Module 1: sMRI Pretrained 3D-ResNet-18 -> Strict Train-Only PCA-5
        # -------------------------------------------------------------
        scaler_m1 = StandardScaler()
        X_m1_tr_s = scaler_m1.fit_transform(X_m1_all[tr_idx])
        X_m1_val_s = scaler_m1.transform(X_m1_all[val_idx])
        
        pca_m1 = PCA(n_components=5, random_state=rep_seed)
        Z_m1_tr = pca_m1.fit_transform(X_m1_tr_s)
        Z_m1_val = pca_m1.transform(X_m1_val_s)
        
        clf_m1 = LogisticRegression(C=0.1, class_weight='balanced', random_state=rep_seed, max_iter=1000)
        clf_m1.fit(Z_m1_tr, y_tr)
        p_m1_tr = clf_m1.predict_proba(Z_m1_tr)[:, 1]
        p_m1_val = clf_m1.predict_proba(Z_m1_val)[:, 1]
        
        # -------------------------------------------------------------
        # Module 2: Questionnaire (9 Transition Items)
        # -------------------------------------------------------------
        # Impute missing values with train medians
        tr_m2_med = np.nanmedian(X_m2_all[tr_idx], axis=0)
        X_m2_tr_imp = np.where(np.isnan(X_m2_all[tr_idx]), tr_m2_med, X_m2_all[tr_idx])
        X_m2_val_imp = np.where(np.isnan(X_m2_all[val_idx]), tr_m2_med, X_m2_all[val_idx])
        
        scaler_m2 = StandardScaler()
        X_m2_tr_s = scaler_m2.fit_transform(X_m2_tr_imp)
        X_m2_val_s = scaler_m2.transform(X_m2_val_imp)
        
        clf_m2 = LogisticRegression(C=0.1, class_weight='balanced', random_state=rep_seed, max_iter=1000)
        clf_m2.fit(X_m2_tr_s, y_tr)
        p_m2_tr = clf_m2.predict_proba(X_m2_tr_s)[:, 1]
        p_m2_val = clf_m2.predict_proba(X_m2_val_s)[:, 1]
        
        # -------------------------------------------------------------
        # Module 3: Phenotypic (4 Transition Items)
        # -------------------------------------------------------------
        tr_m3_med = np.nanmedian(X_m3_all[tr_idx], axis=0)
        X_m3_tr_imp = np.where(np.isnan(X_m3_all[tr_idx]), tr_m3_med, X_m3_all[tr_idx])
        X_m3_val_imp = np.where(np.isnan(X_m3_all[val_idx]), tr_m3_med, X_m3_all[val_idx])
        
        scaler_m3 = StandardScaler()
        X_m3_tr_s = scaler_m3.fit_transform(X_m3_tr_imp)
        X_m3_val_s = scaler_m3.transform(X_m3_val_imp)
        
        clf_m3 = LogisticRegression(C=0.1, class_weight='balanced', random_state=rep_seed, max_iter=1000)
        clf_m3.fit(X_m3_tr_s, y_tr)
        p_m3_tr = clf_m3.predict_proba(X_m3_tr_s)[:, 1]
        p_m3_val = clf_m3.predict_proba(X_m3_val_s)[:, 1]
        
        # -------------------------------------------------------------
        # Trimodal Simple Average (1/3, 1/3, 1/3)
        # -------------------------------------------------------------
        p_avg_tr = (p_m1_tr + p_m2_tr + p_m3_tr) / 3.0
        p_avg_val = (p_m1_val + p_m2_val + p_m3_val) / 3.0
        
        # Helper to compute metrics with training-derived threshold (Zero In-Sample Peeking)
        def eval_fold(p_train_fold, p_val_fold):
            # Optimal threshold derived strictly on training fold via Youden's J
            fpr_tr, tpr_tr, ths_tr = roc_curve(y_tr, p_train_fold)
            opt_th = ths_tr[np.argmax(tpr_tr - fpr_tr)]
            
            auc_v = roc_auc_score(y_val, p_val_fold)
            pr_v = average_precision_score(y_val, p_val_fold)
            
            # Apply training threshold to out-of-fold validation
            preds_v = (p_val_fold >= opt_th).astype(int)
            cm_v = confusion_matrix(y_val, preds_v, labels=[0, 1])
            tn, fp, fn, tp = cm_v.ravel()
            
            se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * prec * se / (prec + se) if (prec + se) > 0 else 0.0
            
            return auc_v, pr_v, f1, se, sp
        
        for mod_key, p_tr_i, p_val_i in [
            ('m1', p_m1_tr, p_m1_val),
            ('m2', p_m2_tr, p_m2_val),
            ('m3', p_m3_tr, p_m3_val),
            ('fusion_avg', p_avg_tr, p_avg_val)
        ]:
            auc_f, pr_f, f1_f, se_f, sp_f = eval_fold(p_tr_i, p_val_i)
            fold_results[mod_key]['auc'].append(auc_f)
            fold_results[mod_key]['pr'].append(pr_f)
            fold_results[mod_key]['f1'].append(f1_f)
            fold_results[mod_key]['sens'].append(se_f)
            fold_results[mod_key]['spec'].append(sp_f)

print(f"Completed {total_folds} folds successfully.")

# 4. Summarize Distribution Across Folds (Non-Parametric & Parametric)
summary_rows = []
model_labels = {
    'm1': 'Module 1: sMRI (Pretrained ResNet-18 + PCA-5)',
    'm2': 'Module 2: Questionnaires (Transition Features)',
    'm3': 'Module 3: Phenotypic (Transition Features)',
    'fusion_avg': 'Trimodal Fusion: Simple Average (33/33/33)'
}

for k, name in model_labels.items():
    aucs = np.array(fold_results[k]['auc'])
    prs = np.array(fold_results[k]['pr'])
    f1s = np.array(fold_results[k]['f1'])
    senss = np.array(fold_results[k]['sens'])
    specs = np.array(fold_results[k]['spec'])
    
    summary_rows.append({
        'Model / Architecture': name,
        'Mean Out-of-Fold AUC': round(float(np.mean(aucs)), 4),
        'SD AUC': round(float(np.std(aucs)), 4),
        'Median AUC': round(float(np.median(aucs)), 4),
        'IQR AUC [25th, 75th]': f"[{np.percentile(aucs, 25):.4f}, {np.percentile(aucs, 75):.4f}]",
        'Range AUC [Min, Max]': f"[{np.min(aucs):.4f}, {np.max(aucs):.4f}]",
        'Mean PR-AUC': round(float(np.mean(prs)), 4),
        'Mean F1': round(float(np.mean(f1s)), 4),
        'Mean Sensitivity': round(float(np.mean(senss)), 4),
        'Mean Specificity': round(float(np.mean(specs)), 4)
    })

df_cv_summary = pd.DataFrame(summary_rows)
df_cv_summary.to_csv(OUT_DIR / 'repeated_family_cv_table.csv', index=False)

print("\n" + "=" * 80)
print("REPEATED FAMILY-STRATIFIED CV (5x5 = 25 FOLDS) SUMMARY TABLE:")
print("=" * 80)
print(df_cv_summary.to_string(index=False))

# Save detailed JSON artifact
json_cv = {
    'validation_protocol': '5 repeats of 5-fold StratifiedGroupKFold on family_id (25 folds)',
    'cohort': {
        'total_participants': int(n_total),
        'incident_cases': int(n_pos),
        'unique_families': int(n_fams),
        'prevalence': float(round(n_pos / n_total, 4))
    },
    'thresholding_policy': 'Optimal Youden threshold fit strictly on training fold, applied out-of-fold',
    'models': summary_rows
}
(OUT_DIR / 'repeated_family_cv_results.json').write_text(json.dumps(json_cv, indent=2), encoding='utf-8')
print("\n07_repeated_family_cv execution complete.")
