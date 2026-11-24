"""
07_repeated_family_cv.py
========================
Repeated Family-Stratified Cross-Validation Pipeline (M6 / Recommendation 1+4)
prev_archi_pro_max (Strict Response to Reviewer 1 & Reviewer 2)

Methodological Compliance:
1. Eliminates single-split bias: Runs 5 repeats of 5-fold StratifiedGroupKFold
   (total 25 distinct family-isolated folds) across the full cohort of 304 participants (152 families).
2. Strict Zero Leakage across Twin Pairs: Twin pairs from the same family (family_id) are strictly
   assigned to the same fold in every single iteration (zero inter-twin contamination).
3. Evaluates all core frameworks across all 25 folds:
   - Module 1: sMRI Pretrained ResNet-18 (Med3D PCA-5)
   - Module 2: Questionnaire Transition Vulnerability (Somatic, Discrepancy, Social Avoidance)
   - Module 3: Phenotypic Transition Vulnerability (Socioeconomic Deprivation, Pubertal Growth Spurt)
   - Trimodal Simple Average Fusion (1/3, 1/3, 1/3)
   - Trimodal Balanced Late Fusion (0.25, 0.45, 0.30)
4. Outputs comprehensive variance and stability tables:
   - results/repeated_family_cv_table.csv
   - results/repeated_family_cv_results.json
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

# 1. Load All 304 Participants with Outcomes
outcomes_df = pd.read_csv(RAW_DIR / 'outcomes.csv')
quest_df = pd.read_csv(RAW_DIR / 'questionnaire.csv')
pheno_df = pd.read_csv(RAW_DIR / 'phenotypic.csv')
mri_data = np.load(MRI_DIR / 'standalone__pretrained.npz', allow_pickle=True)

# Filter outcomes to the 304 cohort participants with complete MRI data
mri_feats_raw = mri_data['features']
mri_pids_list = [str(p) for p in mri_data['pids']]
mri_pid_map = {pid: i for i, pid in enumerate(mri_pids_list)}

outcomes_df = outcomes_df[outcomes_df['participant_id'].astype(str).isin(mri_pid_map)].copy()
all_pids = outcomes_df['participant_id'].astype(str).values
all_fams = outcomes_df['family_id'].fillna('unknown').astype(str).values
all_y = outcomes_df['internalizing_incident'].values.astype(int)

n_total = len(all_pids)
n_pos = sum(all_y)
n_fams = len(np.unique(all_fams))
print(f"Total Cohort: N={n_total} ({n_pos} positive cases, {n_total - n_pos} controls, {n_fams} unique twin families)")

X_mri_all = np.array([mri_feats_raw[mri_pid_map[p]] for p in all_pids], dtype=float)

# Module 2: Questionnaire Transition Features
def extract_m2_all(pids):
    m = pd.DataFrame({'participant_id': pids}).merge(quest_df, on='participant_id', how='left')
    diff_26 = pd.to_numeric(m['pSCAS26'], errors='coerce') - pd.to_numeric(m['SCAS26'], errors='coerce')
    diff_worry = pd.to_numeric(m['pSCAS31'], errors='coerce') - pd.to_numeric(m['SCAS31'], errors='coerce')
    df_feat = pd.DataFrame({
        'pSDQ03': pd.to_numeric(m['pSDQ03'], errors='coerce'),
        'SPHERE07': pd.to_numeric(m['SPHERE07'], errors='coerce'),
        'SPHERE08': pd.to_numeric(m['SPHERE08'], errors='coerce'),
        'SPHERE22': pd.to_numeric(m['SPHERE22'], errors='coerce'),
        'pSCAS31': pd.to_numeric(m['pSCAS31'], errors='coerce'),
        'pSCAS_socpho_score': pd.to_numeric(m['pSCAS_socpho_score'], errors='coerce'),
        'SCAS_socpho_score': pd.to_numeric(m['SCAS_socpho_score'], errors='coerce'),
        'diff_26': diff_26,
        'diff_worry': diff_worry
    })
    return df_feat.values

X_m2_all_raw = extract_m2_all(all_pids)

# Module 3: Phenotypic Transition Features
p_cols = ['birth_IER_decile', 'birth_IRSD_decile', 'birth_IRSAD_decile', 'pPDS01_ses01']
m_p = pd.DataFrame({'participant_id': all_pids}).merge(pheno_df, on='participant_id', how='left')
X_m3_all_raw = m_p[p_cols].apply(pd.to_numeric, errors='coerce').values

print(f"Modality Shapes: M1={X_mri_all.shape}, M2={X_m2_all_raw.shape}, M3={X_m3_all_raw.shape}")

# 3. Repeated Family-Stratified Cross-Validation (5 Repeats x 5 Folds = 25 Folds)
n_repeats = 5
n_splits = 5
total_folds = n_repeats * n_splits

cv_records = {
    'Module 1: sMRI (Pretrained ResNet-18)': {'auc': [], 'pr': [], 'f1': [], 'sens': [], 'spec': []},
    'Module 2: Questionnaires (Transition)': {'auc': [], 'pr': [], 'f1': [], 'sens': [], 'spec': []},
    'Module 3: Phenotypic (Transition)': {'auc': [], 'pr': [], 'f1': [], 'sens': [], 'spec': []},
    'Trimodal Fusion: Simple Average (33/33/33)': {'auc': [], 'pr': [], 'f1': [], 'sens': [], 'spec': []},
    'Trimodal Fusion: Balanced Late (25/45/30)': {'auc': [], 'pr': [], 'f1': [], 'sens': [], 'spec': []}
}

fold_counter = 0
for rep in range(n_repeats):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED + rep * 100)
    
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(all_pids, all_y, all_fams)):
        fold_counter += 1
        
        # Verify zero twin/family overlap
        tr_f = set(all_fams[tr_idx])
        val_f = set(all_fams[val_idx])
        assert len(tr_f.intersection(val_f)) == 0, f"Critical Leakage: Family overlap in fold {fold_counter}!"
        
        y_tr, y_val = all_y[tr_idx], all_y[val_idx]
        
        # 1. Fit Module 1 strictly on training fold
        sc1 = StandardScaler()
        X_mri_tr_sc = sc1.fit_transform(X_mri_all[tr_idx])
        X_mri_val_sc = sc1.transform(X_mri_all[val_idx])
        
        pca1 = PCA(n_components=5, random_state=SEED)
        Z_mri_tr = pca1.fit_transform(X_mri_tr_sc)
        Z_mri_val = pca1.transform(X_mri_val_sc)
        
        clf1 = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=SEED)
        clf1.fit(Z_mri_tr, y_tr)
        p1_val = clf1.predict_proba(Z_mri_val)[:, 1]
        
        # 2. Fit Module 2 strictly on training fold
        m2_tr_med = np.nanmedian(X_m2_all_raw[tr_idx], axis=0)
        X2_tr_imp = np.where(np.isnan(X_m2_all_raw[tr_idx]), m2_tr_med, X_m2_all_raw[tr_idx])
        X2_val_imp = np.where(np.isnan(X_m2_all_raw[val_idx]), m2_tr_med, X_m2_all_raw[val_idx])
        
        sc2 = StandardScaler()
        X2_tr_sc = sc2.fit_transform(X2_tr_imp)
        X2_val_sc = sc2.transform(X2_val_imp)
        
        clf2 = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=SEED)
        clf2.fit(X2_tr_sc, y_tr)
        p2_val = clf2.predict_proba(X2_val_sc)[:, 1]
        
        # 3. Fit Module 3 strictly on training fold
        m3_tr_med = np.nanmedian(X_m3_all_raw[tr_idx], axis=0)
        X3_tr_imp = np.where(np.isnan(X_m3_all_raw[tr_idx]), m3_tr_med, X_m3_all_raw[tr_idx])
        X3_val_imp = np.where(np.isnan(X_m3_all_raw[val_idx]), m3_tr_med, X_m3_all_raw[val_idx])
        
        sc3 = StandardScaler()
        X3_tr_sc = sc3.fit_transform(X3_tr_imp)
        X3_val_sc = sc3.transform(X3_val_imp)
        
        clf3 = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=SEED)
        clf3.fit(X3_tr_sc, y_tr)
        p3_val = clf3.predict_proba(X3_val_sc)[:, 1]
        
        # 4. Multimodal Combinations
        p_avg = (p1_val + p2_val + p3_val) / 3.0
        p_bal = 0.25 * p1_val + 0.45 * p2_val + 0.30 * p3_val
        
        fold_models = {
            'Module 1: sMRI (Pretrained ResNet-18)': p1_val,
            'Module 2: Questionnaires (Transition)': p2_val,
            'Module 3: Phenotypic (Transition)': p3_val,
            'Trimodal Fusion: Simple Average (33/33/33)': p_avg,
            'Trimodal Fusion: Balanced Late (25/45/30)': p_bal
        }
        
        for m_name, p_preds in fold_models.items():
            auc_f = roc_auc_score(y_val, p_preds)
            pr_f = average_precision_score(y_val, p_preds)
            
            # Threshold using Youden's J on fold
            fpr_f, tpr_f, ths_f = roc_curve(y_val, p_preds)
            th_f = ths_f[np.argmax(tpr_f - fpr_f)]
            preds_bin = (p_preds >= th_f).astype(int)
            
            cm_f = confusion_matrix(y_val, preds_bin, labels=[0, 1])
            tn, fp, fn, tp = cm_f.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
            
            cv_records[m_name]['auc'].append(auc_f)
            cv_records[m_name]['pr'].append(pr_f)
            cv_records[m_name]['f1'].append(f1)
            cv_records[m_name]['sens'].append(sens)
            cv_records[m_name]['spec'].append(spec)

print(f"\nAll {total_folds} family-stratified folds completed successfully.")

# 4. Aggregate Performance Across All 25 Folds
cv_summary_rows = []
for m_name, metrics in cv_records.items():
    mean_auc = np.mean(metrics['auc'])
    std_auc = np.std(metrics['auc'])
    ci_auc_low = mean_auc - 1.96 * (std_auc / np.sqrt(total_folds))
    ci_auc_high = mean_auc + 1.96 * (std_auc / np.sqrt(total_folds))
    
    mean_pr = np.mean(metrics['pr'])
    std_pr = np.std(metrics['pr'])
    
    mean_f1 = np.mean(metrics['f1'])
    mean_sens = np.mean(metrics['sens'])
    mean_spec = np.mean(metrics['spec'])
    
    cv_summary_rows.append({
        'Model / Architecture': m_name,
        'Mean Out-of-Fold ROC-AUC': round(mean_auc, 4),
        'SD ROC-AUC': round(std_auc, 4),
        '95% CI ROC-AUC': f"[{ci_auc_low:.4f}, {ci_auc_high:.4f}]",
        'Mean PR-AUC': round(mean_pr, 4),
        'Mean F1': round(mean_f1, 4),
        'Mean Sensitivity': round(mean_sens, 4),
        'Mean Specificity': round(mean_spec, 4)
    })

df_cv = pd.DataFrame(cv_summary_rows)
df_cv.to_csv(OUT_DIR / 'repeated_family_cv_table.csv', index=False)

cv_full_results = {
    'n_repeats': int(n_repeats),
    'n_splits': int(n_splits),
    'total_folds': int(total_folds),
    'cohort_total_n': int(n_total),
    'cohort_pos_n': int(n_pos),
    'unique_families': int(n_fams),
    'summary': cv_summary_rows
}
(OUT_DIR / 'repeated_family_cv_results.json').write_text(json.dumps(cv_full_results, indent=2), encoding='utf-8')

print("\n" + "=" * 80)
print("REPEATED FAMILY-STRATIFIED CV (5 REPEATS x 5 FOLDS = 25 ITERATIONS):")
print("=" * 80)
print(df_cv[['Model / Architecture', 'Mean Out-of-Fold ROC-AUC', 'SD ROC-AUC', '95% CI ROC-AUC', 'Mean PR-AUC', 'Mean F1']].to_string(index=False))
print(f"\n[SAVED] repeated_family_cv_table.csv and repeated_family_cv_results.json")
