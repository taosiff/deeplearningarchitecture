"""
05_shap_analysis.py
===================
Model-Agnostic SHAP Explainability Analysis for Module 2 & Module 3
prev_archi_pro_max (Strict Zero-Leakage & Q1 Publication Standards)

Compliance:
1. Direct Model Evaluation: shap.KernelExplainer runs directly on the genuine
   DeepAttentionProtoNet prediction wrappers (including feature tokenization,
   multi-head self-attention, and learned prototype metric heads).
   No surrogate shortcuts, no simplified dummy functions.
2. Background Distribution: Sampled strictly from the training partition via k-means (k=15).
3. Exact Feature Alignment: Analyzes the exact features for Module 2 (transition vulnerability items)
   and Module 3 (socioeconomic deprivation and pubertal growth spurt).
4. Generates publication-ready 300 DPI figures:
   - Fig 8: Module 2 SHAP Summary Plot
   - Fig 9: Module 2 Feature Importance Plot
   - Fig 10: Module 3 SHAP Summary Plot
   - Fig 11: Module 3 Feature Importance Plot
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
SPLITS_DIR = DATA_DIR / 'splits'
RAW_DIR = DATA_DIR / 'raw'
OUT_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'models'

print("=" * 80)
print("05_SHAP_ANALYSIS: GENUINE DEEP ATTENTION PROTONET INTERPRETABILITY")
print("=" * 80)

# Model Architecture Definition
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, d_token) * 0.05)
        self.biases = nn.Parameter(torch.zeros(n_features, d_token))
        self.feat_emb = nn.Parameter(torch.randn(n_features, d_token) * 0.05)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0) + self.feat_emb.unsqueeze(0)

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_token, n_heads=4, d_ff=64, dropout=0.15):
        super().__init__()
        self.d_token = d_token
        self.n_heads = n_heads
        self.head_dim = d_token // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(d_token, d_token)
        self.k = nn.Linear(d_token, d_token)
        self.v = nn.Linear(d_token, d_token)
        self.proj = nn.Linear(d_token, d_token)
        
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_token),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, D = x.shape
        nx = self.norm1(x)
        q = self.q(nx).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(nx).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(nx).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.proj(out)
        x = x + self.ffn(self.norm2(x))
        return x

class DeepAttentionProtoNet(nn.Module):
    def __init__(self, n_features, d_token=16, n_heads=4, d_embed=16, dropout=0.15):
        super().__init__()
        self.d_token = d_token
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.layer1 = SelfAttentionBlock(d_token, n_heads=n_heads, d_ff=d_token * 4, dropout=dropout)
        self.layer2 = SelfAttentionBlock(d_token, n_heads=n_heads, d_ff=d_token * 4, dropout=dropout)
        
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_token) * 0.05)
        
        self.proj = nn.Sequential(
            nn.Linear(d_token, d_embed),
            nn.LayerNorm(d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed)
        )
        self.temperature = nn.Parameter(torch.tensor([0.2]))

    def forward(self, x):
        B = x.size(0)
        tokens = self.tokenizer(x)
        h = self.layer1(tokens)
        h = self.layer2(h)
        
        scores = torch.bmm(h, self.pool_query.expand(B, -1, -1).transpose(1, 2)) / (self.d_token ** 0.5)
        weights = F.softmax(scores, dim=1)
        pooled = (h * weights).sum(dim=1)
        
        z = self.proj(pooled)
        return F.normalize(z, p=2, dim=-1)

    def compute_prototypes(self, z, y):
        p0 = z[y == 0].mean(dim=0, keepdim=True)
        p1 = z[y == 1].mean(dim=0, keepdim=True)
        return F.normalize(p0, p=2, dim=-1), F.normalize(p1, p=2, dim=-1)

    def predict_logits(self, z, p0, p1):
        tau = self.temperature.clamp(min=0.05, max=1.0)
        sim0 = (z * p0).sum(dim=-1, keepdim=True) / tau
        sim1 = (z * p1).sum(dim=-1, keepdim=True) / tau
        return torch.cat([sim0, sim1], dim=1)

    def predict_proba(self, z, p0, p1):
        logits = self.predict_logits(z, p0, p1)
        return F.softmax(logits, dim=1)[:, 1]

# 1. SHAP Analysis for Module 2 (Questionnaires)
print("\n[1/2] Running SHAP on Module 2 (Questionnaire Deep Attention ProtoNet)...")
m2_checkpoint = torch.load(MODELS_DIR / 'm2_questionnaire_protonet.pt', map_location=device, weights_only=False)
m2_features = m2_checkpoint['feature_names']
p0_m2 = m2_checkpoint['p0'].to(device)
p1_m2 = m2_checkpoint['p1'].to(device)
scaler_m2 = m2_checkpoint['scaler']
train_med_m2 = m2_checkpoint['train_medians']

m2_model = DeepAttentionProtoNet(n_features=len(m2_features), d_token=16, n_heads=4, d_embed=16, dropout=0.15).to(device)
m2_model.load_state_dict(m2_checkpoint['model_state_dict'])
m2_model.eval()

# Load data
train_split = pd.read_csv(SPLITS_DIR / 'train.csv')
test_split = pd.read_csv(SPLITS_DIR / 'test.csv')
quest_df = pd.read_csv(RAW_DIR / 'questionnaire.csv')

def extract_m2_df(split_df):
    m = split_df.merge(quest_df, on='participant_id')
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
    return df_feat

df_m2_tr = extract_m2_df(train_split)
df_m2_te = extract_m2_df(test_split)

X_m2_tr_imp = df_m2_tr.fillna(pd.Series(train_med_m2)).fillna(0).values
X_m2_te_imp = df_m2_te.fillna(pd.Series(train_med_m2)).fillna(0).values

X_m2_tr_sc = scaler_m2.transform(X_m2_tr_imp)
X_m2_te_sc = scaler_m2.transform(X_m2_te_imp)

def m2_predict_wrapper(x_np):
    with torch.no_grad():
        x_t = torch.FloatTensor(x_np).to(device)
        z = m2_model(x_t)
        probs = m2_model.predict_proba(z, p0_m2, p1_m2).cpu().numpy()
    return probs

# Background: K-means on training features (k=15)
bg_m2 = shap.kmeans(X_m2_tr_sc, 15)
explainer_m2 = shap.KernelExplainer(m2_predict_wrapper, bg_m2)

print("Computing SHAP values on test cohort (n=62)...")
shap_values_m2 = explainer_m2.shap_values(X_m2_te_sc, nsamples=100)

mean_abs_m2 = np.mean(np.abs(shap_values_m2), axis=0)
top_m2_idx = np.argsort(mean_abs_m2)[::-1]
top_m2_feats = [m2_features[i] for i in top_m2_idx]

print("\nTop Module 2 Features by SHAP Importance:")
for r, i in enumerate(top_m2_idx, 1):
    print(f"  {r:2d}. {m2_features[i]:25s}: {mean_abs_m2[i]:.4f}")

# Plot Fig 8: Summary Plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
shap.summary_plot(shap_values_m2, X_m2_te_sc, feature_names=m2_features, max_display=10, show=False)
plt.title("Module 2: Behavioral & Symptom SHAP Summary Plot (Test Cohort)", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(OUT_DIR / 'Fig8_Module2_SHAP_Summary.png', dpi=300)
plt.savefig(OUT_DIR / 'Fig8_Module2_SHAP_Summary.tif', dpi=300)
plt.close()
print("[SAVED] Fig8_Module2_SHAP_Summary.png / .tif")

# Plot Fig 9: Feature Importance Bar Plot
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
top_n = len(m2_features)
top_names = [m2_features[i] for i in top_m2_idx[:top_n]][::-1]
top_vals = [mean_abs_m2[i] for i in top_m2_idx[:top_n]][::-1]

y_positions = np.arange(top_n)
ax.barh(y_positions, top_vals, color='#4A90E2', height=0.6, edgecolor='black', linewidth=0.7)
ax.set_yticks(y_positions)
ax.set_yticklabels(top_names, fontsize=10)
ax.set_xlabel("Mean |SHAP Value| (Impact on Model Output Magnitude)", fontsize=11, labelpad=8)
ax.set_title("Module 2: Feature Importance Hierarchy", fontsize=12, pad=12)
for idx, val in enumerate(top_vals):
    ax.text(val + 0.001, idx, f"{val:.4f}", va='center', ha='left', fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / 'Fig9_Module2_Feature_Importance.png', dpi=300)
plt.savefig(OUT_DIR / 'Fig9_Module2_Feature_Importance.tif', dpi=300)
plt.close()
print("[SAVED] Fig9_Module2_Feature_Importance.png / .tif")

# 2. SHAP Analysis for Module 3 (Phenotypic)
print("\n[2/2] Running SHAP on Module 3 (Phenotypic Deep Attention ProtoNet)...")
m3_checkpoint = torch.load(MODELS_DIR / 'm3_static_protonet.pt', map_location=device, weights_only=False)
m3_features = m3_checkpoint['feature_cols']
p0_m3 = m3_checkpoint['p0'].to(device)
p1_m3 = m3_checkpoint['p1'].to(device)
scaler_m3 = m3_checkpoint['scaler']
train_med_m3 = m3_checkpoint['train_medians']

m3_model = DeepAttentionProtoNet(n_features=len(m3_features), d_token=16, n_heads=4, d_embed=16, dropout=0.15).to(device)
m3_model.load_state_dict(m3_checkpoint['model_state_dict'])
m3_model.eval()

pheno_df = pd.read_csv(RAW_DIR / 'phenotypic.csv')
def extract_m3_df(split_df):
    m = split_df.merge(pheno_df, on='participant_id')
    return m[m3_features].apply(pd.to_numeric, errors='coerce')

df_p_tr = extract_m3_df(train_split)
df_p_te = extract_m3_df(test_split)

X_p_tr_imp = df_p_tr.fillna(pd.Series(train_med_m3)).values
X_p_te_imp = df_p_te.fillna(pd.Series(train_med_m3)).values

X_p_tr_sc = scaler_m3.transform(X_p_tr_imp)
X_p_te_sc = scaler_m3.transform(X_p_te_imp)

def m3_predict_wrapper(x_np):
    with torch.no_grad():
        x_t = torch.FloatTensor(x_np).to(device)
        z = m3_model(x_t)
        probs = m3_model.predict_proba(z, p0_m3, p1_m3).cpu().numpy()
    return probs

bg_m3 = shap.kmeans(X_p_tr_sc, 15)
explainer_m3 = shap.KernelExplainer(m3_predict_wrapper, bg_m3)

print("Computing SHAP values on test cohort (n=62)...")
shap_values_m3 = explainer_m3.shap_values(X_p_te_sc, nsamples=100)

mean_abs_m3 = np.mean(np.abs(shap_values_m3), axis=0)
top_m3_idx = np.argsort(mean_abs_m3)[::-1]
top_m3_feats = [m3_features[i] for i in top_m3_idx]

print("\nTop Module 3 Features by SHAP Importance:")
for r, i in enumerate(top_m3_idx, 1):
    print(f"  {r:2d}. {m3_features[i]:30s}: {mean_abs_m3[i]:.4f}")

# Plot Fig 10: Module 3 Summary Plot
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
shap.summary_plot(shap_values_m3, X_p_te_sc, feature_names=m3_features, max_display=10, show=False)
plt.title("Module 3: Phenotypic SHAP Summary Plot (Test Cohort)", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(OUT_DIR / 'Fig10_Module3_SHAP_Summary.png', dpi=300)
plt.savefig(OUT_DIR / 'Fig10_Module3_SHAP_Summary.tif', dpi=300)
plt.close()
print("[SAVED] Fig10_Module3_SHAP_Summary.png / .tif")

# Plot Fig 11: Module 3 Feature Importance Bar Plot
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
top_n = len(m3_features)
top_names_3 = [m3_features[i] for i in top_m3_idx[:top_n]][::-1]
top_vals_3 = [mean_abs_m3[i] for i in top_m3_idx[:top_n]][::-1]

y_positions = np.arange(top_n)
ax.barh(y_positions, top_vals_3, color='#50E3C2', height=0.6, edgecolor='black', linewidth=0.7)
ax.set_yticks(y_positions)
ax.set_yticklabels(top_names_3, fontsize=10)
ax.set_xlabel("Mean |SHAP Value| (Impact on Model Output Magnitude)", fontsize=11, labelpad=8)
ax.set_title("Module 3: Feature Importance Hierarchy", fontsize=12, pad=12)
for idx, val in enumerate(top_vals_3):
    ax.text(val + 0.001, idx, f"{val:.4f}", va='center', ha='left', fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / 'Fig11_Module3_Feature_Importance.png', dpi=300)
plt.savefig(OUT_DIR / 'Fig11_Module3_Feature_Importance.tif', dpi=300)
plt.close()
print("[SAVED] Fig11_Module3_Feature_Importance.png / .tif")

# Save SHAP Summary JSON
shap_results = {
    'module2_top_features': [
        {'rank': r, 'feature': m2_features[i], 'importance': round(float(mean_abs_m2[i]), 4)}
        for r, i in enumerate(top_m2_idx, 1)
    ],
    'module3_top_features': [
        {'rank': r, 'feature': m3_features[i], 'importance': round(float(mean_abs_m3[i]), 4)}
        for r, i in enumerate(top_m3_idx, 1)
    ]
}

(OUT_DIR / 'shap_summary.json').write_text(json.dumps(shap_results, indent=2), encoding='utf-8')
print("[SAVED] shap_summary.json")
print("\nSHAP explainability analysis completed successfully.")
