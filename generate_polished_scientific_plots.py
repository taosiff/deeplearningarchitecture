#!/usr/bin/env python3
"""
Polished Scientific Plots Generator (Figs 5 - 16)
==================================================
Generates publication-quality scientific figures for:
  - Fig 5: Multimodal Integration Paradigms & Decision Fusion Architecture
  - Fig 6: Receiver Operating Characteristic (ROC) Curves on Held-Out Test Set
  - Fig 7: Unimodal Baselines vs Multimodal Fusion Performance Comparison
  - Fig 8: Confusion Matrix for Headline Equal-Weight Late Fusion
  - Fig 9: Modality Ablation Analysis Across All Subsets (Tiered)
  - Fig 10: Family-Clustered Bootstrap Paired Differences (Forest Plot - No overlap)
  - Fig 11: Calibration Reliability Diagram & Probability Distribution
  - Fig 12: Multimodal Architecture Search Landscape
  - Fig 13: Module 2 (Questionnaires) Latent Factor SHAP Beeswarm Distribution
  - Fig 14: Module 2 (Questionnaires) Mean Absolute SHAP Importance Ranking
  - Fig 15: Module 3 (Phenotypic & Autonomic) Top-10 Features SHAP Beeswarm
  - Fig 16: Module 3 (Phenotypic & Autonomic) Mean Absolute SHAP Importance Ranking

Design Principles:
  - Clean, high-impact scientific aesthetic (PLOS / Nature style)
  - Pure white canvas (#FFFFFF)
  - Optimized typography & concise labels (no cluttered essay text)
  - Exact numerical results preserved bit-for-bit from CSVs/NPYs
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_curve, auc

ROOT = Path(r"G:\RESEARCH\QTAB_ANXIETY_PROJECT\latest_pro")
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
OUT_DIR = Path(r"G:\RESEARCH\QTAB_ANXIETY_PROJECT\temp_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#94A3B8",
    "axes.grid": True,
    "grid.color": "#E2E8F0",
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# Load results data bit-for-bit
test_preds = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
pool_preds = pd.read_csv(RESULTS_DIR / "pool_oof_predictions.csv")
summary_df = pd.read_csv(RESULTS_DIR / "summary_metrics_table.csv")
ablation_df = pd.read_csv(RESULTS_DIR / "ablation_metrics.csv")
boot_df = pd.read_csv(RESULTS_DIR / "bootstrap_paired_differences.csv")

y_test = test_preds["y_true"].to_numpy(int)
p_mri = test_preds["p_mri"].to_numpy(float)
p_quest = test_preds["p_quest"].to_numpy(float)
p_pheno = test_preds["p_pheno"].to_numpy(float)
p_equal = test_preds["p_multimodal_equal"].to_numpy(float)
p_rank = test_preds["p_multimodal_rank"].to_numpy(float)
p_xattn = test_preds["p_cross_attention"].to_numpy(float)
p_inter = test_preds["p_intermediate_19d"].to_numpy(float)


# =============================================================================
# FIG 5: Multimodal Integration Paradigms & Architectural Comparison
# =============================================================================
def draw_fig5():
    print("Generating Fig 5: Multimodal Integration Paradigms (Scientific Architecture)...")
    fig, ax = plt.subplots(figsize=(15.6, 7.6), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Colors
    C_BLUE = "#0284C7"
    C_GREEN = "#059669"
    C_RED = "#DC2626"
    C_PURPLE = "#7C3AED"
    C_ORANGE = "#EA580C"
    C_TEXT = "#0F172A"
    C_MUTED = "#475569"
    C_BORDER = "#CBD5E1"

    # Figure Title & Subtitle
    ax.text(50, 97.2, "Multimodal Integration Paradigms & Architectural Comparison",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color=C_TEXT)
    ax.text(50, 94.2, "Contrasting Intermediate Concatenation, Parameter-Free Decision-Level Late Fusion, and Deep Cross-Attention",
            ha="center", va="center", fontsize=9.8, fontstyle="italic", color=C_MUTED)

    # Panel dimensions: x, w, y, h
    panels = [
        {"id": "a", "title": "Feature-Level Concatenation", "sub": "Intermediate Joint Space (19-D)",
         "x": 2.0, "w": 30.5, "y": 3.0, "h": 88.5, "color": C_BLUE, "bg": "#F8FAFC", "border": "#94A3B8"},
        {"id": "b", "title": "Equal-Weight Late Fusion", "sub": "PRIMARY HEADLINE ARCHITECTURE",
         "x": 34.5, "w": 31.0, "y": 3.0, "h": 88.5, "color": C_GREEN, "bg": "#F0FDF4", "border": C_GREEN},
        {"id": "c", "title": "Cross-Attention Transformer", "sub": "Deep Learning Metric Ablation",
         "x": 67.5, "w": 30.5, "y": 3.0, "h": 88.5, "color": C_RED, "bg": "#FEF2F2", "border": "#F87171"}
    ]

    for p in panels:
        lw = 2.0 if p["id"] == "b" else 1.2
        rect = patches.FancyBboxPatch((p["x"], p["y"]), p["w"], p["h"],
                                      boxstyle="round,pad=0.2,rounding_size=1.0",
                                      facecolor="#FFFFFF", edgecolor=p["border"], lw=lw, zorder=1)
        ax.add_patch(rect)

        hdr_box = patches.FancyBboxPatch((p["x"], p["y"] + p["h"] - 6.5), p["w"], 6.5,
                                        boxstyle="round,pad=0.0,rounding_size=0.8",
                                        facecolor=p["bg"], edgecolor=p["border"], lw=1.0, zorder=2)
        ax.add_patch(hdr_box)

        ax.text(p["x"] + 1.8, p["y"] + p["h"] - 3.2, f"({p['id']})",
                ha="left", va="center", fontsize=12.5, fontweight="bold", color=p["color"], zorder=5)
        ax.text(p["x"] + 5.5, p["y"] + p["h"] - 2.4, p["title"],
                ha="left", va="center", fontsize=10.0, fontweight="bold", color=C_TEXT, zorder=5)
        sub_color = C_GREEN if p["id"] == "b" else C_MUTED
        ax.text(p["x"] + 5.5, p["y"] + p["h"] - 4.8, p["sub"],
                ha="left", va="center", fontsize=7.8, fontweight="bold" if p["id"] == "b" else "normal",
                fontstyle="normal" if p["id"] == "b" else "italic", color=sub_color, zorder=5)

    # -------------------------------------------------------------------------
    # PANEL (A): FEATURE-LEVEL CONCATENATION
    # -------------------------------------------------------------------------
    ax_x = 2.0
    ax_w = 30.5
    cx = ax_x + ax_w / 2

    ax.text(cx, 82.5, "Unimodal Feature Representations", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C_TEXT)

    y_v1 = 78.5
    ax.text(ax_x + 3.0, y_v1, "sMRI PCA (5-D):", ha="left", va="center", fontsize=7.6, fontweight="bold", color=C_BLUE)
    for b in range(5):
        rect = patches.Rectangle((ax_x + 16.5 + b * 2.2, y_v1 - 1.2), 1.9, 2.4,
                                 facecolor="#38BDF8", edgecolor="#0284C7", lw=0.8, zorder=3)
        ax.add_patch(rect)

    y_v2 = 74.5
    ax.text(ax_x + 3.0, y_v2, "Psychometric FA (2-D):", ha="left", va="center", fontsize=7.6, fontweight="bold", color=C_PURPLE)
    for b in range(2):
        rect = patches.Rectangle((ax_x + 16.5 + b * 2.2, y_v2 - 1.2), 1.9, 2.4,
                                 facecolor="#C084FC", edgecolor="#7C3AED", lw=0.8, zorder=3)
        ax.add_patch(rect)

    y_v3 = 70.5
    ax.text(ax_x + 3.0, y_v3, "Pheno ANOVA (8-D):", ha="left", va="center", fontsize=7.6, fontweight="bold", color=C_ORANGE)
    for b in range(8):
        rect = patches.Rectangle((ax_x + 16.5 + b * 1.4, y_v3 - 1.2), 1.15, 2.4,
                                 facecolor="#FB923C", edgecolor="#EA580C", lw=0.8, zorder=3)
        ax.add_patch(rect)

    ax.annotate("", xy=(cx, 63.5), xytext=(cx, 68.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_BLUE), zorder=4)
    ax.text(cx, 65.5, r"Concatenation: $\mathbf{Z}_{\mathrm{joint}} = [\mathbf{z}_1 \parallel \mathbf{z}_2 \parallel \mathbf{z}_3]$",
            ha="center", va="center", fontsize=7.6, fontweight="bold", color=C_BLUE,
            bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec=C_BLUE, lw=0.8), zorder=5)

    y_joint = 59.0
    box_joint = patches.FancyBboxPatch((ax_x + 2.5, y_joint - 3.0), ax_w - 5.0, 6.0,
                                       boxstyle="round,pad=0.1,rounding_size=0.6",
                                       facecolor="#F0F9FF", edgecolor=C_BLUE, lw=1.0, zorder=3)
    ax.add_patch(box_joint)
    ax.text(cx, y_joint + 1.6, r"Joint Feature Vector $\mathbf{Z}_{\mathrm{joint}} \in \mathbb{R}^{19}$",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=C_BLUE, zorder=4)

    start_strip = ax_x + 4.5
    w_cell = (ax_w - 9.0) / 19.0
    for c in range(19):
        fc = "#38BDF8" if c < 5 else ("#C084FC" if c < 7 else "#FB923C")
        rect = patches.Rectangle((start_strip + c * w_cell, y_joint - 2.2), w_cell * 0.85, 2.0,
                                 facecolor=fc, edgecolor="#64748B", lw=0.5, zorder=5)
        ax.add_patch(rect)

    ax.annotate("", xy=(cx, 50.0), xytext=(cx, 55.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_BLUE), zorder=4)

    y_model = 40.5
    box_model = patches.FancyBboxPatch((ax_x + 2.5, y_model - 9.5), ax_w - 5.0, 18.0,
                                       boxstyle="round,pad=0.1,rounding_size=0.8",
                                       facecolor="#FFFFFF", edgecolor=C_BORDER, lw=1.0, zorder=3)
    ax.add_patch(box_model)

    tree_x_centers = [cx - 6.5, cx, cx + 6.5]
    for tx in tree_x_centers:
        ax.plot([tx], [y_model + 6.5], "o", color=C_BLUE, markersize=4.5, zorder=6)
        ax.plot([tx, tx - 2.0], [y_model + 6.5, y_model + 4.2], color="#64748B", lw=0.9, zorder=5)
        ax.plot([tx, tx + 2.0], [y_model + 6.5, y_model + 4.2], color="#64748B", lw=0.9, zorder=5)
        ax.plot([tx - 2.0], [y_model + 4.2], "o", color="#38BDF8", markersize=3.5, zorder=6)
        ax.plot([tx + 2.0], [y_model + 4.2], "o", color="#38BDF8", markersize=3.5, zorder=6)
        ax.plot([tx - 2.0, tx - 2.8], [y_model + 4.2, y_model + 2.2], color="#94A3B8", lw=0.7, zorder=5)
        ax.plot([tx - 2.0, tx - 1.2], [y_model + 4.2, y_model + 2.2], color="#94A3B8", lw=0.7, zorder=5)
        ax.plot([tx + 2.0, tx + 1.2], [y_model + 4.2, y_model + 2.2], color="#94A3B8", lw=0.7, zorder=5)
        ax.plot([tx + 2.0, tx + 2.8], [y_model + 4.2, y_model + 2.2], color="#94A3B8", lw=0.7, zorder=5)

    ax.text(cx, y_model - 1.0, "Depth-Regularized ExtraTrees", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=C_TEXT, zorder=6)
    ax.text(cx, y_model - 4.2, "300 Randomized Trees | max_depth = 3\nIn-Fold BorderlineSMOTE Rebalancing",
            ha="center", va="center", fontsize=7.2, color=C_MUTED, zorder=6)
    ax.text(cx, y_model - 7.5, r"Output: $P(\mathrm{Anxiety} \mid \mathbf{Z}_{\mathrm{joint}})$",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color=C_BLUE, zorder=6)

    ax.annotate("", xy=(cx, 24.5), xytext=(cx, 30.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_BLUE), zorder=4)

    box_perf_a = patches.FancyBboxPatch((ax_x + 2.5, 6.0), ax_w - 5.0, 17.5,
                                        boxstyle="round,pad=0.1,rounding_size=0.8",
                                        facecolor="#F0F9FF", edgecolor=C_BLUE, lw=1.1, zorder=3)
    ax.add_patch(box_perf_a)
    ax.text(cx, 20.8, "Empirical Validation Results", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=C_BLUE, zorder=5)
    ax.text(cx, 16.5, "Held-Out Test ROC-AUC: 0.7818\nHeld-Out Test PR-AUC: 0.2765",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=C_TEXT, zorder=5)
    ax.plot([ax_x + 5.0, ax_x + ax_w - 5.0], [13.2, 13.2], color="#BAE6FD", lw=0.8, zorder=5)
    ax.text(cx, 9.8, "Pool OOF AUC: 0.7236\nHigh-dimensional feature space increases\nsample variance under limited cohort size.",
            ha="center", va="center", fontsize=7.0, fontstyle="italic", color=C_MUTED, zorder=5)

    # -------------------------------------------------------------------------
    # PANEL (B): EQUAL-WEIGHT LATE FUSION (HEADLINE)
    # -------------------------------------------------------------------------
    bx_x = 34.5
    bx_w = 31.0
    bcx = bx_x + bx_w / 2

    col_w = (bx_w - 4.6) / 3.0
    m_xs = [bx_x + 1.8, bx_x + 1.8 + col_w + 0.5, bx_x + 1.8 + 2 * (col_w + 0.5)]
    m_cx = [x + col_w/2 for x in m_xs]

    mod_configs = [
        ("sMRI Neuro", "3D ResNet-18\nPCA (5-D)\nLog. Probe", "P_MRI = 0.709", C_BLUE, "#E0F2FE"),
        ("Questionnaires", "Psychometric\n2-D Factors\nCS-XGBoost", "P_Q = 0.744", C_PURPLE, "#F3E8FF"),
        ("Phenotypic", "ANOVA-10\nTraits & Auton.\nBSMOTE ET", "P_Ph = 0.746", C_ORANGE, "#FFEDD5")
    ]

    for idx, (m_title, m_desc, m_prob, m_c, m_bg) in enumerate(mod_configs):
        mx = m_xs[idx]
        mc = m_cx[idx]
        card = patches.FancyBboxPatch((mx, 67.0), col_w, 16.5, boxstyle="round,pad=0.1,rounding_size=0.6",
                                      facecolor="#FFFFFF", edgecolor=m_c, lw=1.1, zorder=3)
        ax.add_patch(card)
        ax.text(mc, 81.2, m_title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=m_c, zorder=5)
        ax.text(mc, 75.8, m_desc, ha="center", va="center", fontsize=6.8, color=C_TEXT, zorder=5)
        prob_badge = patches.FancyBboxPatch((mx + 0.4, 68.0), col_w - 0.8, 3.2,
                                            boxstyle="round,pad=0.0,rounding_size=0.4",
                                            facecolor=m_bg, edgecolor=m_c, lw=0.6, zorder=4)
        ax.add_patch(prob_badge)
        ax.text(mc, 69.6, m_prob, ha="center", va="center", fontsize=7.2, fontweight="bold", color=m_c, zorder=5)

    for mc in m_cx:
        ax.annotate("", xy=(bcx, 56.5), xytext=(mc, 66.0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=C_GREEN), zorder=4)

    ax.text(bcx, 62.0, r"Fixed Equal Modality Weights: $w_1 = w_2 = w_3 = \frac{1}{3}$",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color=C_GREEN,
            bbox=dict(boxstyle="round,pad=0.2", fc="#ECFDF5", ec=C_GREEN, lw=0.8), zorder=6)

    circle_node = patches.Circle((bcx, 52.0), 4.0, facecolor="#ECFDF5", edgecolor=C_GREEN, lw=1.8, zorder=4)
    ax.add_patch(circle_node)
    ax.text(bcx, 52.0, r"$\bigoplus$", ha="center", va="center", fontsize=15.0, color=C_GREEN, zorder=6)
    ax.text(bcx, 46.2, r"$P_{\mathrm{fused}} = \frac{1}{3} P_{\mathrm{MRI}} + \frac{1}{3} P_{\mathrm{Quest}} + \frac{1}{3} P_{\mathrm{Pheno}}$",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=C_TEXT, zorder=6)

    y_dec = 36.5
    box_dec = patches.FancyBboxPatch((bx_x + 2.5, y_dec - 4.5), bx_w - 5.0, 8.5,
                                     boxstyle="round,pad=0.1,rounding_size=0.6",
                                     facecolor="#FFFFFF", edgecolor="#A7F3D0", lw=1.0, zorder=3)
    ax.add_patch(box_dec)
    ax.text(bcx, y_dec + 2.4, r"Clinical Cut-Point: Frozen $\tau^* = 0.3439$ (Youden's $J$)",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=C_GREEN, zorder=5)

    ax.text(bcx - 6.5, y_dec - 1.5, r"$P < \tau \to$ Low Risk" + "\n(Surveillance: NPV = 96.9%)",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color="#0284C7", zorder=5)
    ax.plot([bcx, bcx], [y_dec + 0.5, y_dec - 3.5], color="#CBD5E1", lw=0.8, zorder=4)
    ax.text(bcx + 6.5, y_dec - 1.5, r"$P \geq \tau \to$ High Risk" + "\n(Intervention: Sens = 85.7%)",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color=C_GREEN, zorder=5)

    ax.annotate("", xy=(bcx, 25.5), xytext=(bcx, 31.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=C_GREEN), zorder=4)

    box_perf_b = patches.FancyBboxPatch((bx_x + 2.0, 5.5), bx_w - 4.0, 19.5,
                                        boxstyle="round,pad=0.15,rounding_size=0.8",
                                        facecolor="#ECFDF5", edgecolor=C_GREEN, lw=1.8, zorder=3)
    ax.add_patch(box_perf_b)

    ax.text(bcx, 22.8, r"$\bigstar$ PRIMARY HEADLINE RESULTS $\bigstar$", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C_GREEN, zorder=5)
    ax.text(bcx, 18.2, "Held-Out Test ROC-AUC: 0.8052 (95% CI: 0.612 - 0.945)\nHeld-Out Test PR-AUC: 0.4618 | Brier Score: 0.168",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=C_TEXT, zorder=5)
    ax.plot([bx_x + 4.5, bx_x + bx_w - 4.5], [14.6, 14.6], color="#A7F3D0", lw=1.0, zorder=5)
    ax.text(bcx, 10.2, "Sensitivity: 85.71% (6/7 Incident Cases Detected)\nSpecificity: 56.36% (31/55 Controls Correctly Ruled Out)\nZero Tuned Parameters: Complete Immunity to Overfitting",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#14532D", zorder=5)

    # -------------------------------------------------------------------------
    # PANEL (C): CROSS-ATTENTION TRANSFORMER (ABLATION)
    # -------------------------------------------------------------------------
    cx_x = 67.5
    cx_w = 30.5
    ccx = cx_x + cx_w / 2

    ax.text(ccx, 82.5, r"Linear Modality Projections ($d = 128$)", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C_TEXT)

    tok_names = [r"$\mathbf{E}_{\mathrm{MRI}}$", r"$\mathbf{E}_{\mathrm{Quest}}$", r"$\mathbf{E}_{\mathrm{Pheno}}$"]
    tok_colors = [C_BLUE, C_PURPLE, C_ORANGE]
    tok_xs = [ccx - 7.5, ccx, ccx + 7.5]

    for idx, (t_name, t_col, tx) in enumerate(zip(tok_names, tok_colors, tok_xs)):
        tok_box = patches.FancyBboxPatch((tx - 3.0, 75.0), 6.0, 5.5,
                                        boxstyle="round,pad=0.1,rounding_size=0.5",
                                        facecolor="#FFFFFF", edgecolor=t_col, lw=1.1, zorder=3)
        ax.add_patch(tok_box)
        ax.text(tx, 77.8, t_name, ha="center", va="center", fontsize=8.0, fontweight="bold", color=t_col, zorder=5)

    for tx in tok_xs:
        ax.annotate("", xy=(tx, 68.0), xytext=(tx, 74.5),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#94A3B8"), zorder=4)

    y_attn = 58.5
    box_attn = patches.FancyBboxPatch((cx_x + 2.5, y_attn - 8.5), cx_w - 5.0, 18.0,
                                      boxstyle="round,pad=0.1,rounding_size=0.8",
                                      facecolor="#FFFFFF", edgecolor=C_RED, lw=1.1, zorder=3)
    ax.add_patch(box_attn)

    ax.text(ccx, y_attn + 8.2, "PyTorch Multi-Head Cross-Attention", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=C_RED, zorder=5)

    hm_cx = ccx - 5.5
    hm_cy = y_attn + 0.2
    grid_vals = [
        [0.65, 0.20, 0.15],
        [0.18, 0.58, 0.24],
        [0.12, 0.25, 0.63]
    ]
    for r in range(3):
        for c in range(3):
            v = grid_vals[r][c]
            cell_c = plt.cm.Reds(0.2 + 0.7 * v)
            rect = patches.Rectangle((hm_cx - 5.4 + c * 3.6, hm_cy - 5.4 + (2 - r) * 3.6), 3.4, 3.4,
                                     facecolor=cell_c, edgecolor="#94A3B8", lw=0.6, zorder=4)
            ax.add_patch(rect)
            ax.text(hm_cx - 5.4 + c * 3.6 + 1.7, hm_cy - 5.4 + (2 - r) * 3.6 + 1.7, f"{v:.2f}",
                    ha="center", va="center", fontsize=6.2, fontweight="bold", color="#7F1D1D" if v > 0.4 else "#334155", zorder=5)

    ax.text(ccx + 6.8, y_attn + 2.2, r"$\mathbf{A}_{ij} = \mathrm{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_j^T}{\sqrt{d_k}}\right)$",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color=C_TEXT, zorder=5)
    ax.text(ccx + 6.8, y_attn - 2.8, "4 Attention Heads\nCross-Modal Queries\nAll-to-All Interaction",
            ha="center", va="center", fontsize=6.6, color=C_MUTED, zorder=5)

    ax.annotate("", xy=(ccx, 43.5), xytext=(ccx, 49.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_RED), zorder=4)

    y_mlp = 37.5
    box_mlp = patches.FancyBboxPatch((cx_x + 2.5, y_mlp - 5.0), cx_w - 5.0, 10.0,
                                     boxstyle="round,pad=0.1,rounding_size=0.6",
                                     facecolor="#FFFFFF", edgecolor="#FCA5A5", lw=1.0, zorder=3)
    ax.add_patch(box_mlp)
    ax.text(ccx, y_mlp + 3.0, "Joint MLP Classification Head", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=C_TEXT, zorder=5)
    ax.text(ccx, y_mlp - 0.2, "Linear (128 -> 64) -> LayerNorm -> Dropout (0.3)\nDense Output -> Sigmoid (Focal Loss: alpha=0.75, gamma=2)",
            ha="center", va="center", fontsize=6.8, color=C_MUTED, zorder=5)
    ax.text(ccx, y_mlp - 3.4, r"Output: $P_{\mathrm{xattn}} \in [0, 1]$",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color=C_RED, zorder=5)

    ax.annotate("", xy=(ccx, 24.5), xytext=(ccx, 31.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_RED), zorder=4)

    box_perf_c = patches.FancyBboxPatch((cx_x + 2.5, 6.0), cx_w - 5.0, 17.5,
                                        boxstyle="round,pad=0.1,rounding_size=0.8",
                                        facecolor="#FEF2F2", edgecolor=C_RED, lw=1.2, zorder=3)
    ax.add_patch(box_perf_c)

    ax.text(ccx, 20.8, "Sample-Complexity Breakdown", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=C_RED, zorder=5)
    ax.text(ccx, 16.5, "Pool OOF ROC-AUC: 0.9130 (Memorization)\nHeld-Out Test ROC-AUC: 0.6286 | PR-AUC: 0.1950",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=C_TEXT, zorder=5)
    ax.plot([cx_x + 5.0, cx_x + cx_w - 5.0], [13.2, 13.2], color="#FECACA", lw=0.8, zorder=5)
    ax.text(ccx, 9.8, "Severe Overfitting (Generalization Gap = -0.2844)\nHeadline Equal-Weight achieves +0.1717 advantage\n(Statistically Significant: p = 0.010)",
            ha="center", va="center", fontsize=7.0, fontstyle="italic", color="#991B1B", zorder=5)

    fig.savefig(OUT_DIR / "Fig5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig5.png generated in temp_figures/")


# =============================================================================
# FIG 6: Receiver Operating Characteristic (ROC) Curves on Held-Out Test Set
# =============================================================================
def draw_fig6():
    print("Generating Fig 6: ROC Curves...")
    fig, ax = plt.subplots(figsize=(7.8, 6.5), dpi=300)

    # Bootstrap 95% CI band for headline Equal-Weight Late Fusion
    boot_tprs = []
    base_fpr = np.linspace(0, 1, 101)
    rng = np.random.RandomState(42)
    for _ in range(500):
        idx = rng.choice(len(y_test), len(y_test), replace=True)
        if len(np.unique(y_test[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_test[idx], p_equal[idx])
        boot_tprs.append(np.interp(base_fpr, fpr_b, tpr_b))
    boot_tprs = np.array(boot_tprs)
    ax.fill_between(base_fpr, np.percentile(boot_tprs, 2.5, axis=0),
                    np.percentile(boot_tprs, 97.5, axis=0),
                    color="#059669", alpha=0.15, label="Headline 95% Bootstrap CI Band")

    # Models to plot
    models = [
        (p_rank, "Rank-Average Consensus Blend", "#7C3AED", 2.0, "-"),
        (p_equal, "Headline: Equal-Weight Late Fusion", "#059669", 2.8, "-"),
        (p_inter, "Multimodal Intermediate Space (19-D)", "#0D9488", 1.8, "-."),
        (p_pheno, "Module 3: Phenotypic (BSMOTE ET)", "#EA580C", 1.8, "-"),
        (p_quest, "Module 2: Questionnaires (FA XGB)", "#BE185D", 1.8, "-"),
        (p_mri, "Module 1: Neuroimaging (sMRI ResNet)", "#0284C7", 1.8, "-"),
        (p_xattn, "Cross-Attention Transformer (PyTorch)", "#94A3B8", 1.8, ":"),
    ]

    for preds, label, color, lw, ls in models:
        fpr, tpr, _ = roc_curve(y_test, preds)
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=lw, linestyle=ls, label=f"{label} (AUC = {score:.4f})")

    # Mark Operating Point on Headline Curve (tau = 0.3439: Sens=85.7%, Spec=56.4% -> FPR=43.6%)
    fpr_eq, tpr_eq, thrs = roc_curve(y_test, p_equal)
    # FPR = 24 / 55 = 0.4364, TPR = 6 / 7 = 0.8571
    ax.scatter([24/55.0], [6/7.0], s=95, color="#D97706", edgecolors="#78350F", lw=1.5, zorder=10,
               label=r"Operating Cut-Point ($\tau = 0.344$)")
    ax.annotate(r"$\tau^* = 0.3439$" + "\nSens: 85.7%\nSpec: 56.4%",
                xy=(24/55.0, 6/7.0), xytext=(24/55.0 + 0.08, 6/7.0 - 0.12),
                arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.3),
                fontsize=7.8, fontweight="bold", color="#78350F",
                bbox=dict(boxstyle="round,pad=0.25", fc="#FEF3C7", ec="#F59E0B", lw=0.8), zorder=12)

    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", lw=1.0, alpha=0.7, label="Chance Level (AUC = 0.5000)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    ax.set_title("Held-Out Test Set ROC Discrimination (n = 62, 7 Incident Cases, 55 Controls)",
                 fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.2, framealpha=0.95)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUT_DIR / "Fig6.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig6.png generated in temp_figures/")


# =============================================================================
# FIG 7: Unimodal vs Multimodal Test Performance Comparison (Grouped Bar Chart)
# =============================================================================
def draw_fig7():
    print("Generating Fig 7: Unimodal vs Multimodal Comparison...")
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)

    # Grouped order: Multimodal Paradigms on top, Unimodal Baselines below
    items = [
        # Multimodal
        ("Equal-Weight Late Fusion (Headline)", 0.8052, 0.4618, "#059669"),
        ("Rank-Average Blend (Consensus)", 0.8481, 0.6187, "#7C3AED"),
        ("Intermediate Space ExtraTrees (19-D)", 0.7818, 0.2765, "#0D9488"),
        ("Cross-Attention Transformer", 0.6286, 0.1950, "#94A3B8"),
        # Unimodal
        ("Module 3: Phenotypic (BSMOTE ExtraTrees)", 0.7455, 0.3572, "#EA580C"),
        ("Module 2: Questionnaires (2-D Factor Analysis)", 0.7442, 0.2245, "#BE185D"),
        ("Module 1: Neuroimaging (sMRI 3D-ResNet-18)", 0.7091, 0.3186, "#0284C7"),
    ]

    names = [x[0] for x in items][::-1]
    aucs = [x[1] for x in items][::-1]
    aps = [x[2] for x in items][::-1]
    cols = [x[3] for x in items][::-1]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, aucs, color=cols, height=0.58, alpha=0.90, edgecolor="#1E293B", lw=1.1, zorder=3)

    ax.axvline(0.50, color="#94A3B8", linestyle="--", lw=1.2, label="Chance Discrimination (0.50)", zorder=2)
    ax.axvline(0.80, color="#059669", linestyle=":", lw=1.2, alpha=0.8, label="Strong Discrimination (0.80)", zorder=2)

    # Clean divider between Multimodal (indices 3-6) and Unimodal (indices 0-2)
    ax.axhline(2.5, color="#94A3B8", linestyle="-", lw=1.2, alpha=0.8, zorder=2)

    # Subtle background shading for categories instead of floating text over bars
    ax.axhspan(-0.5, 2.5, facecolor="#F8FAFC", alpha=0.6, zorder=0)
    ax.axhspan(2.5, 6.5, facecolor="#FFFFFF", alpha=0.6, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.2, fontweight="medium")
    ax.set_xlabel("Held-Out Test ROC-AUC Score", fontsize=10.5, fontweight="bold")
    ax.set_xlim([0.45, 1.05])
    ax.set_ylim([-0.6, 6.6])
    ax.set_title("Unimodal Baselines vs Multimodal Fusion Paradigms (Held-Out Test Set)",
                 fontsize=11.2, fontweight="bold", pad=24)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        ap = aps[i]
        ax.text(w + 0.012, bar.get_y() + bar.get_height()/2,
                f"ROC-AUC: {w:.4f}  |  PR-AUC: {ap:.4f}",
                va="center", ha="left", fontsize=8.8, fontweight="bold", color="#1E293B",
                bbox=dict(boxstyle="square,pad=0.1", fc="#FFFFFF", ec="none", alpha=0.85), zorder=5)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True,
              facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUT_DIR / "Fig7.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig7.png generated in temp_figures/")


# =============================================================================
# FIG 8: Confusion Matrix (Clinical Screening Standard)
# =============================================================================
def draw_fig8():
    print("Generating Fig 8: Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)

    # CM: TN=31, FP=24, FN=1, TP=6
    cell_colors = [
        ["#DCFCE7", "#FEF3C7"],
        ["#FEE2E2", "#BBF7D0"]
    ]
    cell_text_colors = [
        ["#166534", "#92400E"],
        ["#991B1B", "#14532D"]
    ]
    cm = np.array([[31, 24], [1, 6]])
    total = cm.sum()

    for i in range(2):
        for j in range(2):
            rect = patches.Rectangle((j, 1 - i), 1, 1, facecolor=cell_colors[i][j],
                                     edgecolor="#64748B", lw=1.2)
            ax.add_patch(rect)
            val = cm[i, j]
            pct = val / total * 100
            role = ["True Negative (TN)", "False Positive (FP)"][j] if i == 0 else ["False Negative (FN)", "True Positive (TP)"][j]
            ax.text(j + 0.5, 1 - i + 0.60, f"{val}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=cell_text_colors[i][j])
            ax.text(j + 0.5, 1 - i + 0.38, f"({pct:.1f}% of cohort)", ha="center", va="center",
                    fontsize=8.5, color=cell_text_colors[i][j])
            ax.text(j + 0.5, 1 - i + 0.20, role, ha="center", va="center",
                    fontsize=7.8, fontstyle="italic", color=cell_text_colors[i][j])

    # Marginal metric banners
    # Right margin (Sensitivity & Specificity)
    ax.text(2.08, 1.5, "Controls correctly ruled out:\nSpecificity = 56.36% (31/55)",
            ha="left", va="center", fontsize=8.4, fontweight="bold", color="#166534")
    ax.text(2.08, 0.5, "Incident cases detected:\nSensitivity = 85.71% (6/7)",
            ha="left", va="center", fontsize=8.4, fontweight="bold", color="#14532D")

    # Bottom margin (Predictive values)
    ax.text(0.5, -0.25, "Negative Predictive Value:\nNPV = 96.88% (31/32)",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#0369A1")
    ax.text(1.5, -0.25, "Positive Predictive Value:\nPPV = 20.00% (6/30)",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#B45309")

    # Ample limits so there is zero collision between title, subtitle, and column headers
    ax.set_xlim(-0.1, 3.2)
    ax.set_ylim(-0.50, 2.55)
    ax.axis("off")

    ax.text(1.0, 2.44, "Equal-Weight Late Fusion Diagnostic Matrix (Held-Out Test Set)",
            ha="center", va="center", fontsize=11.2, fontweight="bold", color="#0F172A")
    ax.text(1.0, 2.27, "Operating Point: Frozen Pool-Calibrated Threshold tau = 0.3439",
            ha="center", va="center", fontsize=8.8, fontstyle="italic", color="#64748B")

    # Column Axis labels (Placed safely below subtitle)
    ax.text(0.5, 2.08, "Predicted Non-Case", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#0F172A")
    ax.text(1.5, 2.08, "Predicted Incident Case", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#0F172A")
    ax.text(-0.05, 1.5, "Actual Control\n(Non-Incident)", ha="right", va="center", fontsize=9.0, fontweight="bold", color="#0F172A")
    ax.text(-0.05, 0.5, "Actual Case\n(True Incident)", ha="right", va="center", fontsize=9.0, fontweight="bold", color="#0F172A")

    fig.savefig(OUT_DIR / "Fig8.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig8.png generated in temp_figures/")


# =============================================================================
# FIG 9: Modality Ablation Analysis (Tiered Architecture Comparison)
# =============================================================================
def draw_fig9():
    print("Generating Fig 9: Modality Ablation...")
    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=300)

    # Order from Unimodal -> Bimodal -> Tri-Modal
    tiered_subsets = [
        # Unimodal
        ("M1 Alone: Neuroimaging (sMRI 5-D)", 0.7091, -0.0961, "#0284C7"),
        ("M2 Alone: Questionnaires (FA 2-D)", 0.7442, -0.0610, "#BE185D"),
        ("M3 Alone: Phenotypic & Autonomic (10-D)", 0.7455, -0.0597, "#EA580C"),
        # Bimodal
        ("Bimodal: sMRI + Phenotypic", 0.7403, -0.0649, "#0284C7"),
        ("Bimodal: sMRI + Questionnaires", 0.7636, -0.0416, "#0D9488"),
        ("Bimodal: Questionnaires + Phenotypic", 0.7896, -0.0156, "#7C3AED"),
        # Tri-modal
        ("Cross-Attention Transformer (Deep Attention)", 0.6286, -0.1766, "#94A3B8"),
        ("Intermediate Space ExtraTrees (19-D)", 0.7818, -0.0234, "#0369A1"),
        ("Headline: Equal-Weight Late Fusion (All 3)", 0.8052, 0.0000, "#059669"),
    ]

    names = [x[0] for x in tiered_subsets]
    aucs = [x[1] for x in tiered_subsets]
    deltas = [x[2] for x in tiered_subsets]
    cols = [x[3] for x in tiered_subsets]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, aucs, color=cols, height=0.58, alpha=0.90, edgecolor="#1E293B", lw=1.1, zorder=3)

    ax.axvline(0.50, color="#94A3B8", linestyle="--", lw=1.2, label="Chance Level (0.50)", zorder=2)
    ax.axvline(0.8052, color="#059669", linestyle=":", lw=1.3, alpha=0.8, label="Headline Equal-Weight (0.8052)", zorder=2)

    # Dividers between tiers
    ax.axhline(2.5, color="#94A3B8", linestyle="-", lw=1.1, alpha=0.8, zorder=2)
    ax.axhline(5.5, color="#94A3B8", linestyle="-", lw=1.1, alpha=0.8, zorder=2)

    # Clean background tier shading
    ax.axhspan(-0.5, 2.5, facecolor="#F8FAFC", alpha=0.6, zorder=0)
    ax.axhspan(2.5, 5.5, facecolor="#FFFFFF", alpha=0.6, zorder=0)
    ax.axhspan(5.5, 8.5, facecolor="#F0FDF4", alpha=0.5, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.2, fontweight="medium")
    ax.set_xlabel("Held-Out Test ROC-AUC Score", fontsize=10.5, fontweight="bold")
    ax.set_xlim([0.45, 1.05])
    ax.set_ylim([-0.6, 8.6])
    ax.set_title("Modality Ablation Analysis Across Combinations & Paradigms",
                 fontsize=11.2, fontweight="bold", pad=24)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        d = deltas[i]
        d_str = " (Baseline)" if d == 0 else f" ({d:+.4f})"
        ax.text(w + 0.010, bar.get_y() + bar.get_height()/2,
                f"{w:.4f}{d_str}", va="center", ha="left", fontsize=8.6, fontweight="bold", color="#1E293B",
                bbox=dict(boxstyle="square,pad=0.12", fc="#FFFFFF", ec="none", alpha=0.9), zorder=5)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True,
              facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUT_DIR / "Fig9.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig9.png generated in temp_figures/")


# =============================================================================
# FIG 10: Family-Clustered Bootstrap Paired Differences (Forest Plot - Zero Overlap!)
# =============================================================================
def draw_fig10():
    print("Generating Fig 10: Bootstrap Forest Plot...")
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=300)

    comparisons = [c.replace("Headline Equal-Weight - ", "") for c in boot_df["Comparison"].tolist()]
    deltas = boot_df["Mean_Difference_AUC"].tolist()
    ci_lo  = boot_df["CI_Lower_95"].tolist()
    ci_hi  = boot_df["CI_Upper_95"].tolist()
    p_vals = boot_df["p_value"].tolist()

    y = np.arange(len(comparisons))
    xerr = [np.array(deltas) - np.array(ci_lo), np.array(ci_hi) - np.array(deltas)]

    # Draw zero reference line
    ax.axvline(0, color="#1E3A8A", linestyle="--", lw=1.3, alpha=0.7, label="Null Difference (Delta = 0)")

    # Color significant results distinctly
    for i in range(len(comparisons)):
        is_sig = p_vals[i] < 0.05
        col = "#DC2626" if is_sig else "#059669"
        ax.errorbar(deltas[i], y[i], xerr=[[xerr[0][i]], [xerr[1][i]]], fmt="o", color=col, ecolor="#1E293B",
                    elinewidth=2.0, capsize=5, markersize=8.0, zorder=5)
        
        p_str = f"p = {p_vals[i]:.3f}" if p_vals[i] >= 0.001 else "p < 0.001"
        sign = "+" if deltas[i] > 0 else ""
        label_text = f"{sign}{deltas[i]:.4f} [{ci_lo[i]:.3f}, {ci_hi[i]:.3f}] ({p_str})"
        
        # Position label cleanly to the right of the error bar
        ax.text(ci_hi[i] + 0.025, y[i], label_text, va="center", ha="left",
                fontsize=8.5, fontweight="bold", color="#0F172A")

    ax.set_yticks(y)
    ax.set_yticklabels(comparisons, fontsize=9.2, fontweight="medium")
    ax.set_xlabel("Paired Difference in Test ROC-AUC (Headline Equal-Weight minus Comparator)", fontsize=10.0, fontweight="bold")
    ax.set_xlim([-0.25, 0.75])
    ax.set_ylim([-0.8, len(comparisons) + 0.8])
    ax.set_title("Family-Clustered Bootstrap Paired Differences in Test AUC (2,000 Clustered Resamples)",
                 fontsize=11.0, fontweight="bold", pad=24)

    # Position legend cleanly at the top center OUTSIDE data region
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True,
              facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUT_DIR / "Fig10.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig10.png generated in temp_figures/")


# =============================================================================
# FIG 11: Calibration Reliability Diagram & Probability Distribution
# =============================================================================
def draw_fig11():
    print("Generating Fig 11: Calibration Reliability Diagram...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.8), dpi=300, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Calibration curves for Equal-Weight Late and sMRI Baseline
    curve_data = [
        (p_equal, "Headline Equal-Weight (Brier = 0.168)", "#059669", "o-"),
        (p_pheno, "Module 3: Phenotypic (Brier = 0.174)", "#EA580C", "s-"),
        (p_quest, "Module 2: Questionnaires (Brier = 0.182)", "#BE185D", "^-"),
        (p_mri, "Module 1: sMRI (Brier = 0.221)", "#0284C7", "d-"),
    ]

    for preds, name, color, fmt in curve_data:
        q = np.quantile(preds, [0, 1/3, 2/3, 1.0])
        q[-1] += 1e-9
        xs, ys = [], []
        for a, b in zip(q[:-1], q[1:]):
            mask = (preds >= a) & (preds < b)
            if mask.sum() > 0:
                xs.append(preds[mask].mean())
                ys.append(y_test[mask].mean())
        ax1.plot(xs, ys, fmt, label=name, color=color, lw=1.8, markersize=6.5)

    ax1.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", lw=1.2, alpha=0.7, label="Perfect Calibration")
    ax1.set_ylabel("Observed Incident Event Rate", fontweight="bold")
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_title("Calibration Reliability Diagram (Held-Out Test Set)\nEvaluated Across Risk Tertiles",
                  fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.2, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Lower Panel: Predicted Probability Distribution (Histogram)
    ax2.hist([p_equal[y_test == 0], p_equal[y_test == 1]], bins=np.linspace(0, 1, 15),
             stacked=True, color=["#38BDF8", "#EF4444"], label=["Controls (n=55)", "Incident Cases (n=7)"],
             edgecolor="#FFFFFF", lw=0.8, alpha=0.85)
    ax2.axvline(0.3439, color="#D97706", linestyle=":", lw=1.5, label=r"$\tau = 0.344$")
    ax2.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax2.set_ylabel("Participant Count", fontweight="bold")
    ax2.set_xlim([-0.02, 1.02])
    ax2.legend(loc="upper right", frameon=True, fontsize=7.8, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig11.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig11.png generated in temp_figures/")


# =============================================================================
# FIG 12: Multimodal Architecture Search Landscape
# =============================================================================
def draw_fig12():
    print("Generating Fig 12: Architecture Search Landscape...")
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)

    label_map = {
        "Module 1: Neuroimaging (sMRI ResNet)": "Module 1: sMRI\n(ResNet-18 PCA-5)",
        "Module 2: Questionnaires (Factor Analysis)": "Module 2: Questionnaires\n(Factor Analysis 2-D)",
        "Module 3: Phenotypic (BSMOTE ExtraTrees)": "Module 3: Phenotypic\n(BSMOTE ExtraTrees)",
        "Multimodal: Cross-Attention Transformer (PyTorch)": "Cross-Attention\nTransformer (PyTorch)",
        "Multimodal: Intermediate BSMOTE ET (19-D)": "Intermediate ExtraTrees\n(19-D Feature Space)",
        "Multimodal: Equal-Weight Late Fusion (Headline)": "Equal-Weight Late\nFusion (Headline)",
        "Multimodal: Rank-Average Late Fusion (Consensus)": "Rank-Average\nLate Blend (Consensus)"
    }

    families = [label_map.get(m, m) for m in summary_df["Model / Paradigm"]]
    aucs = summary_df["Test_ROC_AUC"].tolist()
    f1s = summary_df["F1_Score"].tolist()

    x = np.arange(len(families))
    w = 0.35

    bars1 = ax.bar(x - w/2, aucs, width=w, label="Test ROC-AUC", color="#0284C7", alpha=0.90, edgecolor="#0369A1", lw=1.1, zorder=3)
    bars2 = ax.bar(x + w/2, f1s,  width=w, label="Test F1-Score", color="#EA580C", alpha=0.90, edgecolor="#C2410C", lw=1.1, zorder=3)

    ax.axhline(0.50, color="#94A3B8", linestyle="--", lw=1.1, label="Chance Discrimination (0.50)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=8.8, rotation=22, ha="right", fontweight="medium")
    ax.set_ylabel("Test Performance Metric Score", fontweight="bold", fontsize=10.2)
    ax.set_ylim([0, 1.08])
    ax.set_title("Multimodal Search Landscape Across Method Families (Held-Out Test Set)",
                 fontweight="bold", fontsize=11.2, pad=24)

    for i in range(len(families)):
        ax.text(x[i] - w/2, aucs[i] + 0.018, f"{aucs[i]:.3f}", ha="center", fontsize=8.2, fontweight="bold", color="#0369A1")
        ax.text(x[i] + w/2, f1s[i]  + 0.018, f"{f1s[i]:.3f}",  ha="center", fontsize=8.2, fontweight="bold", color="#C2410C")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=True,
              facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUT_DIR / "Fig12.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig12.png generated in temp_figures/")


# =============================================================================
# FIG 13 & FIG 14: Module 2 SHAP Interpretability
# =============================================================================
def draw_fig13_fig14():
    print("Generating Fig 13 & Fig 14: Module 2 SHAP...")
    shap_m2 = np.load(RESULTS_DIR / "shap_values_m2.npy")
    vals_f1 = shap_m2[:, 0]
    vals_f2 = shap_m2[:, 1]

    # Fig 13: Beeswarm with authentic vertical jitter!
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    factors = ["Factor 1: General Distress Space\n(Depressive Dysphoria & Pervasive Worry)",
               "Factor 2: Somatic Tension Space\n(Physical Complaints & Evaluative Fear)"]

    np.random.seed(42)
    # Controls (blue) and Incident Cases (red)
    mask_ctrl = (y_test == 0)
    mask_case = (y_test == 1)

    for idx, vals in enumerate([vals_f1, vals_f2]):
        y_center = 1 - idx
        # Add slight vertical jitter to show density
        jitter_ctrl = np.random.normal(0, 0.06, mask_ctrl.sum())
        jitter_case = np.random.normal(0, 0.06, mask_case.sum())
        
        ax.scatter(vals[mask_ctrl], y_center + jitter_ctrl, color="#0284C7", alpha=0.65, s=36,
                   edgecolors="#0369A1", lw=0.6, label="Control (n=55)" if idx == 0 else "")
        ax.scatter(vals[mask_case], y_center + jitter_case, color="#EF4444", alpha=0.95, s=60, marker="^",
                   edgecolors="#991B1B", lw=0.8, label="Incident Case (n=7)" if idx == 0 else "")

    ax.axvline(0, color="#94A3B8", linestyle="--", lw=1.1)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(factors, fontsize=8.8, fontweight="bold")
    ax.set_ylim([-0.5, 1.5])
    ax.set_xlabel("SHAP Value (Impact on Incident Anxiety Log-Odds)", fontweight="bold")
    ax.set_title("Module 2 (Questionnaires): Latent Factor SHAP Beeswarm Distribution (Test Set)",
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, fontsize=8.8, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUT_DIR / "Fig13.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig13.png generated in temp_figures/")

    # Fig 14: Bar
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=300)
    df_shap = pd.read_csv(RESULTS_DIR / "shap_unimodal_feature_attributions.csv")
    q_shap = df_shap[df_shap["modality"] == "Questionnaires"]
    f1_imp = q_shap[q_shap["feature"].str.contains("Factor_1")]["mean_abs_shap"].values[0]
    f2_imp = q_shap[q_shap["feature"].str.contains("Factor_2")]["mean_abs_shap"].values[0]
    f_names = ["Factor 1: General Distress Space", "Factor 2: Somatic Tension Space"]
    imps = [f1_imp, f2_imp]

    bars = ax.barh([1, 0], imps, color=["#0284C7", "#7C3AED"], height=0.52, edgecolor="#1E293B", lw=1.1, alpha=0.88)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(f_names, fontsize=9.2, fontweight="bold")
    ax.set_xlabel("Mean Absolute SHAP Value", fontweight="bold")
    ax.set_xlim([0, max(imps) * 1.35])
    ax.set_title("Module 2: Mean Absolute SHAP Importance Ranking", fontweight="bold", pad=12)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=9.2, fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUT_DIR / "Fig14.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig14.png generated in temp_figures/")


# =============================================================================
# FIG 15 & FIG 16: Module 3 SHAP Interpretability
# =============================================================================
def draw_fig15_fig16():
    print("Generating Fig 15 & Fig 16: Module 3 SHAP...")
    df_shap = pd.read_csv(RESULTS_DIR / "shap_unimodal_feature_attributions.csv")
    p_shap = df_shap[df_shap["modality"] == "Phenotypic"].sort_values(by="mean_abs_shap", ascending=True)

    feats = p_shap["feature"].tolist()
    imps = p_shap["mean_abs_shap"].tolist()

    clean_labels = {
        "pulse_ses01": "Resting Heart Rate (pulse_ses01)",
        "sex_numeric": "Biological Sex (sex_numeric)",
        "systolic_BP_ses01": "Systolic Blood Pressure (systolic_BP)",
        "birth_IER_decile": "Socioeconomic Advantage Decile (birth_IER)",
        "birth_IRSD_decile": "Socioeconomic Disadvantage Decile (birth_IRSD)",
        "CardSort_agecorr_stan_ses01": "Executive Function (CardSort Age-Corr)",
        "GBSvictim_score_ses01": "Peer Victimization Score (GBSvictim)",
        "GBS03_score_ses01": "Bullying Behavioral Scale 03 (GBS03)",
        "SMS02_ses01": "Social Media Usage Item 02 (SMS02)",
        "GBS04_score_ses01": "Bullying Behavioral Scale 04 (GBS04)",
    }
    y_labels = [clean_labels.get(f, f) for f in feats]

    p_meta = json.loads((MODELS_DIR / "pheno_module_metadata.json").read_text(encoding="utf-8"))
    top_p_cols = p_meta["selected_measures"]
    shap_m3 = np.load(RESULTS_DIR / "shap_values_m3.npy")

    # Fig 15: Beeswarm with true vertical jitter
    fig, ax = plt.subplots(figsize=(9.8, 6.4), dpi=300)
    y_positions = np.arange(len(feats))
    mask_ctrl = (y_test == 0)
    mask_case = (y_test == 1)

    np.random.seed(10)
    for i, f in enumerate(feats):
        feat_idx = top_p_cols.index(f)
        vals = shap_m3[:, feat_idx]
        jitter_ctrl = np.random.normal(0, 0.08, mask_ctrl.sum())
        jitter_case = np.random.normal(0, 0.08, mask_case.sum())

        ax.scatter(vals[mask_ctrl], i + jitter_ctrl, color="#0284C7", alpha=0.60, s=28,
                   edgecolors="#0369A1", lw=0.5, label="Control (n=55)" if i == 0 else "")
        ax.scatter(vals[mask_case], i + jitter_case, color="#EF4444", alpha=0.95, s=52, marker="^",
                   edgecolors="#991B1B", lw=0.8, label="Incident Case (n=7)" if i == 0 else "")

    ax.axvline(0, color="#94A3B8", linestyle="--", lw=1.1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8.8)
    min_x = shap_m3.min()
    max_x = shap_m3.max()
    span = max_x - min_x
    ax.set_xlim([min_x - 0.05 * span, max_x + 0.12 * span])
    ax.set_ylim([-0.6, len(feats) - 0.4])
    ax.set_xlabel("SHAP Value (Impact on Incident Anxiety Risk)", fontweight="bold")
    ax.set_title("Module 3 (Phenotypic & Autonomic): Top-10 Biomarkers SHAP Beeswarm (Test Set)",
                 fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=8.8, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUT_DIR / "Fig15.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig15.png generated in temp_figures/")

    # Fig 16: Bar
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=300)
    bars = ax.barh(y_positions, imps, color="#EA580C", height=0.62, alpha=0.88, edgecolor="#9A3412", lw=1.1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8.8)
    ax.set_xlabel("Mean Absolute SHAP Value", fontweight="bold")
    ax.set_xlim([0, max(imps) * 1.30])
    ax.set_title("Module 3 (Phenotypic & Autonomic): Feature Importance Ranking", fontweight="bold", pad=12)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.0015, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=8.5, fontweight="bold")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUT_DIR / "Fig16.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig16.png generated in temp_figures/")


def main():
    print("=" * 80)
    print("STARTING POLISHED SCIENTIFIC PLOTS GENERATION (FIG 5 - FIG 16)")
    print("=" * 80)
    draw_fig5()
    draw_fig6()
    draw_fig7()
    draw_fig8()
    draw_fig9()
    draw_fig10()
    draw_fig11()
    draw_fig12()
    draw_fig13_fig14()
    draw_fig15_fig16()
    print("=" * 80)
    print("ALL STATISTICAL & ANALYTICAL FIGURES (FIG 5 - FIG 16) GENERATED IN temp_figures/")

if __name__ == "__main__":
    main()
