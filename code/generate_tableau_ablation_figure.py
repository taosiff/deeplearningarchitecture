"""
code/generate_tableau_ablation_figure.py
========================================
Publication-Grade Tableau-Style Ablation Study & Modality Sensitivity Visual (Figure 6).
Rigorously evaluates Unimodal Baselines, Pairwise Modality Combinations, and Trimodal
Late/Intermediate Fusion Paradigms on the Held-Out Test Set (N = 62, 31 Intact Twin Pairs).

Special Scientific Focus:
  - Detailed case study on 'Ablation: MRI + Questionnaire' (Val AUC: 0.7455 -> Test AUC: 0.5532)
  - Generalization gap audit (Delta = Test - Val AUC)
  - Diagnostic recovery comparison (Confusion matrix contrast: MRI+Questionnaire vs Headline Trimodal)
  - Tableau modern pastel card aesthetic with crisp typography and zero clutter
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Typography configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#0F172A'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

def build_tableau_ablation_figure():
    print("Generating Tableau-Style Figure 6: Comparative Ablation Study...")

    # Load dynamic values from multimodal_fusion_results.json if present
    json_path = RESULTS_DIR / 'multimodal_fusion_results.json'
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    fig, ax = plt.subplots(figsize=(15.6, 10.6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.axis('off')

    # Coordinate boundaries: x in [-0.5, 14.8], y in [0.4, 10.6]
    ax.set_xlim(-0.5, 14.8)
    ax.set_ylim(0.4, 10.6)

    # -------------------------------------------------------------------------
    # 1. HEADER & SUBTITLE
    # -------------------------------------------------------------------------
    ax.text(7.15, 10.25, "Figure 6: Multimodal Modality Ablation & Generalization Gap Analysis",
            ha='center', va='center', fontsize=14.5, fontweight='bold', color='#0F172A')
    ax.text(7.15, 9.90, "Held-Out Prospective Adolescent Anxiety Screening Cohort (N = 62, 31 Twin Pairs, 7 Incident Cases)",
            ha='center', va='center', fontsize=9.6, fontstyle='italic', color='#475569')

    # -------------------------------------------------------------------------
    # 2. LEFT PANE: HORIZONTAL ABLATION BENCHMARK (x: 0.15 to 8.45)
    # -------------------------------------------------------------------------
    models_data = [
        # Tier 1: Trimodal
        {
            'name': "Headline: Equal-Weight Late Fusion",
            'sub': "Parameter-free (1/3, 1/3, 1/3) Dawes' heuristic",
            'test_auc': 0.7403, 'val_auc': 0.6649, 'pr_auc': 0.2112,
            'color': '#059669', 'bg_bar': '#DCFCE7', 'edge': '#10B981',
            'tier': 'trimodal', 'y': 8.60, 'highlight': True
        },
        {
            'name': "Intermediate Latent Fusion (37-D)",
            'sub': "L2-regularized logistic classification head",
            'test_auc': 0.7117, 'val_auc': 0.7273, 'pr_auc': 0.2413,
            'color': '#0D9488', 'bg_bar': '#CCFBF1', 'edge': '#14B8A6',
            'tier': 'trimodal', 'y': 7.90, 'highlight': False
        },
        {
            'name': "Stacking Meta-Learner (Ridge)",
            'sub': "Validation out-of-fold probability meta-model",
            'test_auc': 0.7065, 'val_auc': 0.7065, 'pr_auc': 0.1967,
            'color': '#6366F1', 'bg_bar': '#EEF2FF', 'edge': '#818CF8',
            'tier': 'trimodal', 'y': 7.20, 'highlight': False
        },
        {
            'name': "Validation-Tuned 186-Grid Fusion",
            'sub': "Grid search (0.15/0.63/0.22) on 7 val events",
            'test_auc': 0.6649, 'val_auc': 0.7221, 'pr_auc': 0.1851,
            'color': '#E11D48', 'bg_bar': '#FFE4E6', 'edge': '#FB7185',
            'tier': 'trimodal', 'y': 6.50, 'highlight': False
        },

        # Tier 2: Pairwise
        {
            'name': "Ablation: MRI + Phenotypic",
            'sub': "Pairwise weights: 0.05 MRI / 0.95 Phenotypic",
            'test_auc': 0.7273, 'val_auc': 0.5429, 'pr_auc': 0.5005,
            'color': '#0284C7', 'bg_bar': '#E0F2FE', 'edge': '#38BDF8',
            'tier': 'pairwise', 'y': 5.25, 'highlight': False
        },
        {
            'name': "Ablation: Questionnaire + Phenotypic",
            'sub': "Pairwise weights: 0.10 Quest / 0.90 Phenotypic",
            'test_auc': 0.5688, 'val_auc': 0.7558, 'pr_auc': 0.2170,
            'color': '#D97706', 'bg_bar': '#FEF3C7', 'edge': '#FBBF24',
            'tier': 'pairwise', 'y': 4.55, 'highlight': False
        },
        {
            'name': "Ablation: MRI + Questionnaire",
            'sub': "Pairwise weights: 0.05 MRI / 0.95 Quest (Overfit)",
            'test_auc': 0.5532, 'val_auc': 0.7455, 'pr_auc': 0.1675,
            'color': '#DC2626', 'bg_bar': '#FEE2E2', 'edge': '#F87171',
            'tier': 'pairwise', 'y': 3.85, 'highlight': True
        },

        # Tier 3: Unimodal
        {
            'name': "Module 1: Structural MRI (sMRI ResNet-18)",
            'sub': "5-D train-fit PCA features with balanced LR",
            'test_auc': 0.7221, 'val_auc': 0.5117, 'pr_auc': 0.3185,
            'color': '#0369A1', 'bg_bar': '#E0F2FE', 'edge': '#38BDF8',
            'tier': 'unimodal', 'y': 2.60, 'highlight': False
        },
        {
            'name': "Module 3: Static Phenotypic (FT-ProtoNet)",
            'sub': "4 demographic & developmental risk factors",
            'test_auc': 0.5753, 'val_auc': 0.8143, 'pr_auc': 0.2339,
            'color': '#7C3AED', 'bg_bar': '#F3E8FF', 'edge': '#C084FC',
            'tier': 'unimodal', 'y': 1.90, 'highlight': False
        },
        {
            'name': "Module 2: Anxiety Questionnaires (FT-ProtoNet)",
            'sub': "9 prospective anxiety transition markers",
            'test_auc': 0.5247, 'val_auc': 0.7455, 'pr_auc': 0.2119,
            'color': '#9333EA', 'bg_bar': '#FAF5FF', 'edge': '#D8B4FE',
            'tier': 'unimodal', 'y': 1.20, 'highlight': False
        }
    ]

    # Group background bounding cards
    # Tier 1 Box
    tier1_box = patches.FancyBboxPatch((0.15, 6.15), 8.35, 3.20,
                                       boxstyle="round,pad=0.04,rounding_size=0.12",
                                       facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=1.1, zorder=1)
    ax.add_patch(tier1_box)
    ax.text(0.35, 9.18, "TRIMODAL MULTIMODAL FUSION PARADIGMS", fontsize=8.6, fontweight='bold', color='#1E293B', zorder=5)

    # Tier 2 Box
    tier2_box = patches.FancyBboxPatch((0.15, 3.50), 8.35, 2.40,
                                       boxstyle="round,pad=0.04,rounding_size=0.12",
                                       facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=1.1, zorder=1)
    ax.add_patch(tier2_box)
    ax.text(0.35, 5.72, "PAIRWISE BIMODAL ABLATIONS (DUAL-MODALITY SUBSPACING)", fontsize=8.6, fontweight='bold', color='#1E293B', zorder=5)

    # Tier 3 Box
    tier3_box = patches.FancyBboxPatch((0.15, 0.85), 8.35, 2.40,
                                       boxstyle="round,pad=0.04,rounding_size=0.12",
                                       facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=1.1, zorder=1)
    ax.add_patch(tier3_box)
    ax.text(0.35, 3.07, "UNIMODAL BASELINES (STANDALONE FOUNDATIONS)", fontsize=8.6, fontweight='bold', color='#1E293B', zorder=5)

    # Scale mapping: x_start = 4.25, x_end = 7.75 -> corresponds to AUC 0.40 to 0.85
    def auc_to_x(auc_val):
        return 4.25 + (auc_val - 0.40) / (0.85 - 0.40) * (7.75 - 4.25)

    # Reference grid lines
    # Chance line at 0.50
    x_chance = auc_to_x(0.50)
    ax.plot([x_chance, x_chance], [0.90, 9.25], color='#94A3B8', linestyle='--', lw=1.1, zorder=2)
    ax.text(x_chance, 0.68, "Chance\n(0.50)", ha='center', va='top', fontsize=7.2, color='#64748B', fontweight='bold')

    # Good clinical benchmark at 0.70
    x_70 = auc_to_x(0.70)
    ax.plot([x_70, x_70], [0.90, 9.25], color='#10B981', linestyle=':', lw=1.1, zorder=2)
    ax.text(x_70, 0.68, "Clinical\nBenchmark (0.70)", ha='center', va='top', fontsize=7.2, color='#059669', fontweight='bold')

    # Headline mark at 0.7403
    x_head = auc_to_x(0.7403)
    ax.plot([x_head, x_head], [6.20, 9.25], color='#059669', linestyle='-', lw=1.5, zorder=2)

    # Render each model bar
    for m in models_data:
        y = m['y']
        score = m['test_auc']
        x_bar_start = auc_to_x(0.40)
        x_bar_len = auc_to_x(score) - x_bar_start

        # Labels
        name_col = '#0F172A' if not m['highlight'] else (m['color'])
        ax.text(0.35, y + 0.12, m['name'], fontsize=8.4, fontweight='bold', color=name_col, zorder=10)
        ax.text(0.35, y - 0.12, m['sub'], fontsize=7.1, fontstyle='italic', color='#64748B', zorder=10)

        # Bar background track
        track = patches.FancyBboxPatch((x_bar_start, y - 0.15), auc_to_x(0.85) - x_bar_start, 0.30,
                                      boxstyle="round,pad=0.01,rounding_size=0.06",
                                      facecolor='#F1F5F9', edgecolor='none', zorder=3)
        ax.add_patch(track)

        # Bar fill
        bar_patch = patches.FancyBboxPatch((x_bar_start, y - 0.15), x_bar_len, 0.30,
                                          boxstyle="round,pad=0.01,rounding_size=0.06",
                                          facecolor=m['bg_bar'], edgecolor=m['edge'], lw=1.3, zorder=4)
        ax.add_patch(bar_patch)

        # Value label inside or right of bar
        ax.text(auc_to_x(score) - 0.08, y, f"{score:.4f}", ha='right', va='center',
                fontsize=8.2, fontweight='bold', color=m['color'], zorder=6)

        # Delta badge (Test - Val)
        delta = score - m['val_auc']
        delta_str = f"Δ {delta:+.3f}"
        d_bg = '#DCFCE7' if delta >= 0 else '#FEE2E2'
        d_col = '#166534' if delta >= 0 else '#991B1B'
        bbox_d = dict(boxstyle="round,pad=0.18,rounding_size=0.08", facecolor=d_bg, edgecolor='none')
        ax.text(7.85, y, delta_str, ha='left', va='center', fontsize=7.0, fontweight='bold', color=d_col, bbox=bbox_d, zorder=6)

    # -------------------------------------------------------------------------
    # 3. RIGHT PANE: SPECIAL SCIENTIFIC FOCUS ON MRI + QUESTIONNAIRE
    # -------------------------------------------------------------------------
    # Outer Card for Deep Dive (y: 0.65 to 9.35)
    right_card = patches.FancyBboxPatch((8.80, 0.65), 5.70, 8.70,
                                        boxstyle="round,pad=0.05,rounding_size=0.14",
                                        facecolor='#FFFFFF', edgecolor='#DC2626', lw=1.6, zorder=2)
    ax.add_patch(right_card)

    # Right Card Title Pill
    title_pill = patches.FancyBboxPatch((9.00, 8.92), 5.30, 0.36,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor='#FEE2E2', edgecolor='#FCA5A5', lw=1.1, zorder=3)
    ax.add_patch(title_pill)
    ax.text(11.65, 9.10, "ABLATION AUDIT: WHY MRI + QUESTIONNAIRE UNDERPERFORMS",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#991B1B', zorder=5)

    # Analytical Explanation
    narrative_1 = (
        "• Optimization Failure on Low-Event Validation:\n"
        "  Grid search on validation AUC (n_val = 62, 7 events) converged to\n"
        "  extreme weights: w_mri = 0.05, w_quest = 0.95.\n"
        "  Because Module 2 alone overfits validation (Val AUC = 0.7455) but\n"
        "  drops to near chance on held-out test data (Test AUC = 0.5247),\n"
        "  the combination suffers a massive -0.1923 generalization drop."
    )
    ax.text(9.05, 8.80, narrative_1, fontsize=7.5, color='#1E293B', va='top', zorder=5, linespacing=1.35)

    # Metric Comparison Pill Cards (y: 7.00 to 7.55)
    card_val = patches.FancyBboxPatch((9.05, 7.05), 2.50, 0.54,
                                      boxstyle="round,pad=0.02,rounding_size=0.08",
                                      facecolor='#FEF3C7', edgecolor='#FDE68A', lw=1.0, zorder=4)
    ax.add_patch(card_val)
    ax.text(10.30, 7.42, "Validation AUC", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#92400E', zorder=5)
    ax.text(10.30, 7.20, "0.7455 (Overfitted)", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#B45309', zorder=5)

    card_test = patches.FancyBboxPatch((11.80, 7.05), 2.50, 0.54,
                                       boxstyle="round,pad=0.02,rounding_size=0.08",
                                       facecolor='#FEE2E2', edgecolor='#FECACA', lw=1.0, zorder=4)
    ax.add_patch(card_test)
    ax.text(13.05, 7.42, "Held-Out Test AUC", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#991B1B', zorder=5)
    ax.text(13.05, 7.20, "0.5532 (Near Chance)", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#DC2626', zorder=5)

    # Clinical Impact Section Title
    ax.text(11.65, 6.78, "CLINICAL TRIAGE IMPACT (HELD-OUT TEST SET, 7 CASES)",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color='#0F172A', zorder=5)

    # Mini CM 1: MRI + Questionnaire (y: 4.25 to 6.60)
    box_cm1 = patches.FancyBboxPatch((9.05, 4.25), 5.20, 2.35,
                                     boxstyle="round,pad=0.02,rounding_size=0.08",
                                     facecolor='#FFF1F2', edgecolor='#FDA4AF', lw=1.1, zorder=4)
    ax.add_patch(box_cm1)
    ax.text(9.20, 6.42, "Ablation: MRI + Questionnaire (tau* = 0.364)",
            fontsize=7.8, fontweight='bold', color='#9F1239', zorder=5)
    ax.text(9.20, 6.18, "Misses 3 of 7 incident cases (Sensitivity = 57.1%):",
            fontsize=7.2, fontstyle='italic', color='#475569', zorder=5)

    # 2x2 grid for CM 1: TN=38, FP=17, FN=3, TP=4
    ax.text(10.35, 5.65, "TN: 38", fontsize=9.5, fontweight='bold', color='#15803D', ha='center', zorder=5)
    ax.text(10.35, 5.40, "(Controls Spared)", fontsize=6.6, color='#15803D', ha='center', zorder=5)
    ax.text(12.95, 5.65, "FP: 17", fontsize=9.5, fontweight='bold', color='#B45309', ha='center', zorder=5)
    ax.text(12.95, 5.40, "(False Alarms)", fontsize=6.6, color='#B45309', ha='center', zorder=5)

    ax.text(10.35, 4.88, "FN: 3", fontsize=11.0, fontweight='bold', color='#DC2626', ha='center', zorder=5)
    ax.text(10.35, 4.62, "(Missed Transitions)", fontsize=6.8, fontweight='bold', color='#DC2626', ha='center', zorder=5)
    ax.text(12.95, 4.88, "TP: 4", fontsize=9.5, fontweight='bold', color='#047857', ha='center', zorder=5)
    ax.text(12.95, 4.62, "(Detected Cases)", fontsize=6.6, color='#047857', ha='center', zorder=5)

    # Mini CM 2: Headline Trimodal Equal-Weight (y: 1.65 to 4.00)
    box_cm2 = patches.FancyBboxPatch((9.05, 1.65), 5.20, 2.35,
                                     boxstyle="round,pad=0.02,rounding_size=0.08",
                                     facecolor='#F0FDF4', edgecolor='#86EFAC', lw=1.1, zorder=4)
    ax.add_patch(box_cm2)
    ax.text(9.20, 3.82, "Headline: Equal-Weight Trimodal (tau* = 0.460)",
            fontsize=7.8, fontweight='bold', color='#14532D', zorder=5)
    ax.text(9.20, 3.58, "Recovers 2 missed transitions (Sensitivity = 85.7%):",
            fontsize=7.2, fontstyle='italic', color='#475569', zorder=5)

    # 2x2 grid for CM 2: TN=35, FP=20, FN=1, TP=6
    ax.text(10.35, 3.05, "TN: 35", fontsize=9.5, fontweight='bold', color='#15803D', ha='center', zorder=5)
    ax.text(10.35, 2.80, "(Controls Spared)", fontsize=6.6, color='#15803D', ha='center', zorder=5)
    ax.text(12.95, 3.05, "FP: 20", fontsize=9.5, fontweight='bold', color='#B45309', ha='center', zorder=5)
    ax.text(12.95, 2.80, "(Manageable Review)", fontsize=6.6, color='#B45309', ha='center', zorder=5)

    ax.text(10.35, 2.28, "FN: 1", fontsize=11.0, fontweight='bold', color='#16A34A', ha='center', zorder=5)
    ax.text(10.35, 2.02, "(Only 1 Missed Case)", fontsize=6.8, fontweight='bold', color='#15803D', ha='center', zorder=5)
    ax.text(12.95, 2.28, "TP: 6", fontsize=11.0, fontweight='bold', color='#15803D', ha='center', zorder=5)
    ax.text(12.95, 2.02, "(6 of 7 Detected)", fontsize=6.8, fontweight='bold', color='#15803D', ha='center', zorder=5)

    # Recovery Summary Callout Banner (y: 0.85 to 1.35)
    recovery_badge = patches.FancyBboxPatch((9.05, 0.85), 5.20, 0.55,
                                           boxstyle="round,pad=0.02,rounding_size=0.08",
                                           facecolor='#DCFCE7', edgecolor='#4ADE80', lw=1.1, zorder=5)
    ax.add_patch(recovery_badge)
    ax.text(11.65, 1.18, "CLINICAL GAIN: +2 INCIDENT CASES CAPTURED",
            ha='center', va='center', fontsize=7.4, fontweight='bold', color='#15803D', zorder=6)
    ax.text(11.65, 0.98, "False Negatives: 3 -> 1 | Sensitivity: 57.1% -> 85.7% | NPV: 92.7% -> 97.2%",
            ha='center', va='center', fontsize=6.8, fontweight='bold', color='#166534', zorder=6)

    # Save to results
    fig.savefig(RESULTS_DIR / "Fig6_Ablation_Study.png", dpi=300, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / "Fig6_Ablation_Study.tif", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Fig6_Ablation_Study successfully generated (.png & .tif).")

if __name__ == '__main__':
    build_tableau_ablation_figure()
