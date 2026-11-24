"""
code/generate_publication_architectures.py
=========================================
Perfected Publication-Grade 3D Scientific Architecture Figures (Figs 1, 2, 3, 4)
Logical, Mathematical, and Aesthetic Perfection for Q1 Journal Review

Generates:
  - Fig 1: Overall Multimodal Framework (Primary Simple Average Baseline + Calibrated Integration)
  - Fig 2: Module 1 — Structural Neuroimaging Pipeline (Med3D 3D-ResNet-18 + Train-Only PCA-5 + Balanced LR)
  - Fig 3: Module 2 — Longitudinal Anxiety Symptomatology Pipeline (FT-ProtoNet on 9 Transition Markers)
  - Fig 4: Detailed Deep Feature-Tokenizer Prototypical Network Architecture (Intra-Sample Attention + Metric Head)

Aesthetic & Technical Standards:
  - Pure white background (#FFFFFF), high contrast, crisp vectors
  - Clean isometric 3D beveled blocks with top highlight and right shading
  - Generous canvas padding and perfect margin offsets to eliminate any clipping
  - Icon-text separation: All text has dedicated margin from vector icons (zero collisions)
  - Curved residual connections and 3D prototype spheres with radial highlights
  - All metrics and formulas clearly contained within cards
  - 300 DPI high-resolution output (.png and .tif)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Typography settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#0F172A'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'

# -------------------------------------------------------------------------
# CORE 3D DRAWING HELPERS
# -------------------------------------------------------------------------
def draw_3d_block(ax, x, y, w, h, depth=0.35, angle=35,
                  face_col='#1F618D', top_col='#5DADE2', side_col='#154360',
                  edge_col='#0E2F44', lw=1.1, zorder=10):
    """Draws an isometric 3D block with top highlight and right shading."""
    rad = np.radians(angle)
    dx = depth * np.cos(rad)
    dy = depth * np.sin(rad)
    
    # Front face
    front = patches.Rectangle((x, y), w, h, facecolor=face_col,
                              edgecolor=edge_col, linewidth=lw, zorder=zorder)
    ax.add_patch(front)
    
    # Top face (Highlight)
    top_pts = np.array([[x, y + h], [x + w, y + h], [x + w + dx, y + h + dy], [x + dx, y + h + dy]])
    top = patches.Polygon(top_pts, closed=True, facecolor=top_col,
                          edgecolor=edge_col, linewidth=lw, zorder=zorder+1)
    ax.add_patch(top)
    
    # Right side face (Shading)
    side_pts = np.array([[x + w, y], [x + w + dx, y + dy], [x + w + dx, y + h + dy], [x + w, y + h]])
    side = patches.Polygon(side_pts, closed=True, facecolor=side_col,
                           edgecolor=edge_col, linewidth=lw, zorder=zorder+1)
    ax.add_patch(side)
    return dx, dy

def draw_pill(ax, x, y, w, h, text='', face_col='#FFFFFF', edge_col='#94A3B8',
              text_col='#0F172A', font_size=8.5, font_weight='bold', zorder=20):
    """Draws a clean rounded card / pill container with centered text."""
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.14",
                                 facecolor=face_col, edgecolor=edge_col, linewidth=1.1, zorder=zorder)
    ax.add_patch(box)
    if text:
        ax.text(x + w/2.0, y + h/2.0, text, ha='center', va='center',
                color=text_col, fontsize=font_size, fontweight=font_weight, zorder=zorder+5)

def draw_arrow(ax, x1, y1, x2, y2, color='#64748B', lw=1.6, badge='', badge_col='#0284C7', zorder=15):
    """Draws a crisp directional arrow with optional weight badge."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=14),
                zorder=zorder)
    if badge:
        bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bbox_p = dict(boxstyle="round,pad=0.25,rounding_size=0.2", facecolor=badge_col,
                      edgecolor='none', alpha=0.95)
        ax.text(bx, by, badge, ha='center', va='center', color='white',
                fontsize=8.0, fontweight='bold', bbox=bbox_p, zorder=zorder+5)

def draw_sphere(ax, cx, cy, r=0.28, base_col='#10B981', highlight_col='#6EE7B7', edge_col='#047857', zorder=30):
    """Draws a pseudo-3D shaded sphere with realistic radial lighting."""
    # Base circle
    ax.add_patch(patches.Circle((cx, cy), r, facecolor=base_col, edgecolor=edge_col, lw=1.2, zorder=zorder))
    # Specular highlight
    ax.add_patch(patches.Circle((cx - r*0.35, cy + r*0.35), r*0.45,
                                facecolor=highlight_col, edgecolor='none', alpha=0.75, zorder=zorder+1))
    ax.add_patch(patches.Circle((cx - r*0.40, cy + r*0.40), r*0.20,
                                facecolor='white', edgecolor='none', alpha=0.90, zorder=zorder+2))

# -------------------------------------------------------------------------
# VECTOR ICONS
# -------------------------------------------------------------------------
def draw_brain_vector(ax, cx, cy, s=0.26, col='white', zorder=35):
    """Draws a crisp anatomical brain vector silhouette."""
    for dx, dy in [(-s*0.35, 0), (s*0.35, 0), (-s*0.25, s*0.3), (s*0.25, s*0.3)]:
        ax.add_patch(patches.Circle((cx + dx, cy + dy), s*0.36, facecolor='none', edgecolor=col, lw=1.3, zorder=zorder))
    ax.plot([cx, cx], [cy - s*0.4, cy + s*0.5], color=col, lw=1.4, zorder=zorder)
    ax.plot([cx - s*0.28, cx - s*0.08], [cy + s*0.1, cy + s*0.18], color=col, lw=1.1, zorder=zorder)
    ax.plot([cx + s*0.08, cx + s*0.28], [cy + s*0.1, cy + s*0.18], color=col, lw=1.1, zorder=zorder)

def draw_clipboard_vector(ax, cx, cy, s=0.26, col='white', zorder=35):
    """Draws a crisp clinical clipboard vector with checklist."""
    w, h = s * 0.85, s * 1.15
    ax.add_patch(patches.Rectangle((cx - w/2, cy - h/2), w, h, facecolor='none', edgecolor=col, lw=1.3, zorder=zorder))
    ax.add_patch(patches.Rectangle((cx - w*0.3, cy + h*0.38), w*0.6, h*0.18, facecolor=col, edgecolor=col, lw=0.8, zorder=zorder))
    for dy in [-0.25, 0.05, 0.32]:
        ax.plot([cx - w*0.12, cx + w*0.3], [cy + dy*h*0.65, cy + dy*h*0.65], color=col, lw=1.1, zorder=zorder)
        ax.plot([cx - w*0.35, cx - w*0.25, cx - w*0.18], [cy + dy*h*0.65, cy + dy*h*0.65 - 0.04, cy + dy*h*0.65 + 0.05],
                color=col, lw=1.1, zorder=zorder)

def draw_user_vector(ax, cx, cy, s=0.26, col='white', zorder=35):
    """Draws a crisp participant / demographic avatar vector."""
    ax.add_patch(patches.Circle((cx, cy + s*0.38), s*0.30, facecolor='none', edgecolor=col, lw=1.3, zorder=zorder))
    ax.add_patch(patches.Arc((cx, cy - s*0.48), s*1.15, s*1.15, angle=0, theta1=35, theta2=145, edgecolor=col, lw=1.3, zorder=zorder))
    ax.plot([cx - s*0.35, cx - s*0.15, cx, cx + s*0.15, cx + s*0.35],
            [cy - s*0.2, cy - s*0.2, cy - s*0.05, cy - s*0.35, cy - s*0.2], color=col, lw=1.1, zorder=zorder)


# =========================================================================
# FIGURE 1: OVERALL MULTIMODAL PREDICTION FRAMEWORK
# =========================================================================
def build_fig1():
    print("Generating Figure 1: Overall Multimodal Architecture Framework...")
    fig, ax = plt.subplots(figsize=(16.8, 9.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 16.8)
    ax.set_ylim(-0.4, 9.8)
    ax.axis('off')

    # Top Modules (3D Beveled Blocks)
    # Module 1 (Teal/Navy - Structural Neuroimaging)
    draw_3d_block(ax, 0.7, 7.3, 4.5, 1.45, depth=0.35, angle=35,
                  face_col='#1F618D', top_col='#5DADE2', side_col='#154360', edge_col='#0B2535', zorder=10)
    ax.add_patch(patches.Circle((1.35, 8.02), 0.38, facecolor='#154360', edgecolor='#5DADE2', lw=1.0, zorder=25))
    draw_brain_vector(ax, 1.35, 8.02, s=0.26, col='#EBF5FB', zorder=30)
    ax.text(2.0, 8.20, "Module 1: STRUCTURAL MRI", ha='left', va='center',
            color='white', fontsize=10.5, fontweight='bold', zorder=30)
    ax.text(2.0, 7.86, "Med3D ResNet-18 + Train-Only PCA-5", ha='left', va='center',
            color='#D4E6F1', fontsize=8.2, zorder=30)
    ax.text(2.0, 7.56, "Balanced Logistic Regression (C=0.1)", ha='left', va='center',
            color='#A9CCE3', fontsize=7.6, zorder=30)

    # Module 2 (Emerald Green - Behavioral Questionnaires)
    draw_3d_block(ax, 6.15, 7.3, 4.5, 1.45, depth=0.35, angle=35,
                  face_col='#239B56', top_col='#58D68D', side_col='#196F3D', edge_col='#0E4B24', zorder=10)
    ax.add_patch(patches.Circle((6.80, 8.02), 0.38, facecolor='#196F3D', edgecolor='#58D68D', lw=1.0, zorder=25))
    draw_clipboard_vector(ax, 6.80, 8.02, s=0.25, col='#E8F8F5', zorder=30)
    ax.text(7.45, 8.20, "Module 2: QUESTIONNAIRES", ha='left', va='center',
            color='white', fontsize=10.5, fontweight='bold', zorder=30)
    ax.text(7.45, 7.86, "FT-ProtoNet (9 Transition Markers)", ha='left', va='center',
            color='#D1F2EB', fontsize=8.2, zorder=30)
    ax.text(7.45, 7.56, "Intra-Sample Attention (Zero Inversion)", ha='left', va='center',
            color='#A3E4D7', fontsize=7.6, zorder=30)

    # Module 3 (Warm Rust/Orange - Phenotypic & Demographics)
    draw_3d_block(ax, 11.6, 7.3, 4.5, 1.45, depth=0.35, angle=35,
                  face_col='#D35400', top_col='#F39C12', side_col='#935116', edge_col='#592804', zorder=10)
    ax.add_patch(patches.Circle((12.25, 8.02), 0.38, facecolor='#935116', edgecolor='#F39C12', lw=1.0, zorder=25))
    draw_user_vector(ax, 12.25, 8.02, s=0.25, col='#FEF5E7', zorder=30)
    ax.text(12.90, 8.20, "Module 3: PHENOTYPIC", ha='left', va='center',
            color='white', fontsize=10.5, fontweight='bold', zorder=30)
    ax.text(12.90, 7.86, "FT-ProtoNet (4 SES & Pubertal Features)", ha='left', va='center',
            color='#FDEBD0', fontsize=8.2, zorder=30)
    ax.text(12.90, 7.56, "Intra-Sample Attention (Robust Scaler)", ha='left', va='center',
            color='#FAD7A0', fontsize=7.6, zorder=30)

    # Downward inference flow arrows
    draw_arrow(ax, 2.95, 7.3, 2.95, 6.25, color='#64748B', lw=1.8, zorder=12)
    draw_arrow(ax, 8.40, 7.3, 8.40, 6.25, color='#64748B', lw=1.8, zorder=12)
    draw_arrow(ax, 13.85, 7.3, 13.85, 6.25, color='#64748B', lw=1.8, zorder=12)
    ax.text(2.95, 6.75, "Sample-Independent Scoring", ha='center', va='center',
            color='#475569', fontsize=8.0, fontstyle='italic', zorder=25)
    ax.text(8.40, 6.75, "Sample-Independent Scoring", ha='center', va='center',
            color='#475569', fontsize=8.0, fontstyle='italic', zorder=25)
    ax.text(13.85, 6.75, "Sample-Independent Scoring", ha='center', va='center',
            color='#475569', fontsize=8.0, fontstyle='italic', zorder=25)

    # Middle Row: Unimodal Probability Slabs (Silver/Grey 3D Beveled Bars)
    draw_3d_block(ax, 1.15, 5.5, 3.6, 0.72, depth=0.25, angle=35,
                  face_col='#F1F5F9', top_col='#FFFFFF', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(2.95, 5.86, "P_MRI  (Test AUC: 0.722)", ha='center', va='center',
            color='#0F172A', fontsize=10.2, fontweight='bold', zorder=30)

    draw_3d_block(ax, 6.60, 5.5, 3.6, 0.72, depth=0.25, angle=35,
                  face_col='#F1F5F9', top_col='#FFFFFF', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(8.40, 5.86, "P_Quest  (Test AUC: 0.525)", ha='center', va='center',
            color='#0F172A', fontsize=10.2, fontweight='bold', zorder=30)

    draw_3d_block(ax, 12.05, 5.5, 3.6, 0.72, depth=0.25, angle=35,
                  face_col='#F1F5F9', top_col='#FFFFFF', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(13.85, 5.86, "P_Pheno  (Test AUC: 0.575)", ha='center', va='center',
            color='#0F172A', fontsize=10.2, fontweight='bold', zorder=30)

    # Clean diagonal arrows with prominent w = 1/3 weight badges (No text-line collisions!)
    draw_arrow(ax, 2.95, 5.5, 6.25, 4.15, color='#64748B', lw=1.8, badge='w = 1/3', badge_col='#1F618D', zorder=12)
    draw_arrow(ax, 8.40, 5.5, 8.40, 4.15, color='#64748B', lw=1.8, badge='w = 1/3', badge_col='#239B56', zorder=12)
    draw_arrow(ax, 13.85, 5.5, 10.55, 4.15, color='#64748B', lw=1.8, badge='w = 1/3', badge_col='#D35400', zorder=12)

    # Multimodal Decision-Level Fusion 3D Bar (Royal Purple)
    draw_3d_block(ax, 3.4, 2.85, 10.0, 1.25, depth=0.35, angle=35,
                  face_col='#6C3483', top_col='#A569BD', side_col='#4A235A', edge_col='#2C1337', zorder=10)
    ax.text(8.40, 3.70, "Primary Decision-Level Fusion (Simple Average)", ha='center', va='center',
            color='white', fontsize=11.8, fontweight='bold', zorder=30)
    ax.text(8.40, 3.32, "P_multimodal = (1/3) * P_MRI + (1/3) * P_Quest + (1/3) * P_Pheno",
            ha='center', va='center', color='#F5EEF8', fontsize=9.6, fontweight='bold', zorder=30)
    ax.text(8.40, 3.00, "Dawes' Uniform Prior — Minimax Robustness Against Overfitting in Finite-Sample Regimes",
            ha='center', va='center', color='#E8DAEF', fontsize=8.0, fontstyle='italic', zorder=30)

    # Downward arrow to Threshold Calibration
    draw_arrow(ax, 8.40, 2.85, 8.40, 2.15, color='#64748B', lw=1.6, zorder=12)
    ax.text(8.40, 2.50, "Frozen Cut-Point from Validation", ha='center', va='center',
            color='#64748B', fontsize=8.0, fontstyle='italic', zorder=25)

    # Threshold Optimization 3D Bar (Rich Gold / Amber)
    draw_3d_block(ax, 4.6, 1.45, 7.6, 0.70, depth=0.25, angle=35,
                  face_col='#D4AC0D', top_col='#F4D03F', side_col='#9A7D0A', edge_col='#4D3E05', zorder=10)
    ax.text(8.40, 1.88, "Validation-Derived Decision Threshold (tau* = 0.484, Youden's J)",
            ha='center', va='center', color='white', fontsize=10.2, fontweight='bold', zorder=30)
    ax.text(8.40, 1.60, "Threshold Frozen Exclusively from Validation Cohort (Zero Test Reuse)",
            ha='center', va='center', color='#FEF9E7', fontsize=8.0, zorder=30)

    # Downward arrow to Final Prediction
    draw_arrow(ax, 8.40, 1.45, 8.40, 0.85, color='#64748B', lw=1.6, zorder=12)

    # Final Clinical Decision 3D Bar (Teal / Deep Cyan)
    draw_3d_block(ax, 3.5, -0.20, 9.8, 1.05, depth=0.30, angle=35,
                  face_col='#148F77', top_col='#48C9B0', side_col='#0E6252', edge_col='#083E34', zorder=10)
    ax.text(8.40, 0.58, "Prospective Incident Anxiety Prediction (Held-Out Test)", ha='center', va='center',
            color='white', fontsize=11.5, fontweight='bold', zorder=30)
    ax.text(8.40, 0.28, "ROC-AUC = 0.7403 (95% CI: 0.605 - 0.887) | F1 = 0.3636 | Sensitivity = 85.7% (6/7)",
            ha='center', va='center', color='white', fontsize=9.0, fontweight='bold', zorder=30)
    ax.text(8.40, 0.02, "Specificity = 63.6% (35/55) | NPV = 97.2% | Prevalence-Adjusted PPV = 25.3% | NNS = 9.3",
            ha='center', va='center', color='#E8F8F5', fontsize=8.4, zorder=30)

    fig.savefig(OUT_DIR / "Fig1_Overall_Framework.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / "Fig1_Overall_Framework.tif", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Fig1_Overall_Framework.png & .tif successfully generated.")


# =========================================================================
# FIGURE 2: MODULE 1 — STRUCTURAL NEUROIMAGING PIPELINE
# =========================================================================
def build_fig2():
    print("Generating Figure 2: Module 1 Structural MRI Architecture...")
    fig, ax = plt.subplots(figsize=(16.8, 9.0), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0.2, 9.2)
    ax.axis('off')

    # 1. INPUT RAW SCANS (Stacked dark cards on Left)
    draw_3d_block(ax, 0.35, 4.4, 2.1, 2.8, depth=0.25, angle=35,
                  face_col='#1E293B', top_col='#334155', side_col='#0F172A', edge_col='#020617', zorder=10)
    ax.text(1.40, 6.65, "Input Raw Scans\nT1-weighted MRI\n(91 x 109 x 91)", ha='center', va='center',
            color='white', fontsize=8.8, fontweight='bold', zorder=30)

    # Stylized axial brain cross-section inside input block
    for r in [0.48, 0.38, 0.28]:
        ax.add_patch(patches.Ellipse((1.40, 5.20), r*1.5, r*1.9, angle=0,
                                     facecolor='none', edgecolor='#64748B', lw=1.0, linestyle='--', zorder=25))
    ax.add_patch(patches.Ellipse((1.27, 5.25), 0.16, 0.40, angle=15, facecolor='#0284C7', alpha=0.6, zorder=26))
    ax.add_patch(patches.Ellipse((1.53, 5.25), 0.16, 0.40, angle=-15, facecolor='#0284C7', alpha=0.6, zorder=26))
    ax.text(1.40, 4.58, "N = 304 Subjects\n(152 Twin Pairs)", ha='center', va='center',
            color='#94A3B8', fontsize=7.5, zorder=30)

    draw_arrow(ax, 2.65, 5.80, 3.05, 5.80, color='#475569', lw=1.8, zorder=15)

    # 2. PREPROCESSING PIPELINE (Slate Blue Panel)
    panel_prep = patches.FancyBboxPatch((3.05, 0.8), 3.5, 8.0, boxstyle="round,pad=0.15,rounding_size=0.2",
                                        facecolor='#344955', edgecolor='#232F34', lw=1.4, zorder=5)
    ax.add_patch(panel_prep)
    ax.text(4.80, 8.32, "Standardized MRI Preprocessing\n(FSL / ANTs Equivalent)", ha='center', va='center',
            color='white', fontsize=9.8, fontweight='bold', zorder=25)

    prep_steps = [
        ("Skull Stripping (Otsu/BET)", 7.0, "skull"),
        ("N4 Bias Field Correction", 5.5, "pulse"),
        ("MNI152 Non-linear Registration", 4.0, "brain"),
        ("Intensity Z-score Normalization", 2.5, "chart")
    ]
    for title, py, itype in prep_steps:
        pill = patches.FancyBboxPatch((3.20, py - 0.52), 3.2, 1.05, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor='#ECEFF1', edgecolor='#CFD8DC', lw=1.0, zorder=7)
        ax.add_patch(pill)
        # Dedicated icon column on left (x = 3.55), text starts at x = 4.05 (left-aligned, zero overlap!)
        if itype == 'skull':
            ax.add_patch(patches.Circle((3.55, py), 0.20, facecolor='#263238', edgecolor='none', zorder=15))
            ax.add_patch(patches.Circle((3.49, py+0.04), 0.05, facecolor='white', zorder=16))
            ax.add_patch(patches.Circle((3.61, py+0.04), 0.05, facecolor='white', zorder=16))
        elif itype == 'pulse':
            ax.plot([3.35, 3.45, 3.51, 3.58, 3.65, 3.75], [py, py, py+0.24, py-0.24, py+0.12, py],
                    color='#0284C7', lw=1.8, zorder=15)
        elif itype == 'brain':
            ax.add_patch(patches.Circle((3.55, py), 0.20, facecolor='none', edgecolor='#0D9488', lw=1.6, zorder=15))
            ax.plot([3.55, 3.55], [py-0.18, py+0.18], color='#0D9488', lw=1.2, zorder=16)
        elif itype == 'chart':
            ax.add_patch(patches.Rectangle((3.41, py-0.18), 0.08, 0.26, facecolor='#64748B', zorder=15))
            ax.add_patch(patches.Rectangle((3.52, py-0.18), 0.08, 0.38, facecolor='#64748B', zorder=15))
            ax.add_patch(patches.Rectangle((3.63, py-0.18), 0.08, 0.18, facecolor='#64748B', zorder=15))

        ax.text(4.02, py, title, ha='left', va='center', color='#1E293B', fontsize=8.0, fontweight='bold', zorder=25)

    draw_arrow(ax, 6.70, 5.80, 7.10, 5.80, color='#475569', lw=1.8, zorder=15)

    # 3. MED3D RESNET-18 BACKBONE (Large Center Panel)
    panel_cnn = patches.FancyBboxPatch((7.10, 0.8), 5.7, 8.0, boxstyle="round,pad=0.15,rounding_size=0.2",
                                       facecolor='#344955', edgecolor='#232F34', lw=1.4, zorder=5)
    ax.add_patch(panel_cnn)
    ax.text(9.95, 8.32, "Med3D 3D-ResNet-18 Backbone\n(Pretrained Volumetric Embeddings)", ha='center', va='center',
            color='white', fontsize=10.2, fontweight='bold', zorder=25)

    # Upper inner box for feature map tensor slabs
    inner_box = patches.FancyBboxPatch((7.30, 4.4), 5.3, 3.3, boxstyle="round,pad=0.1,rounding_size=0.15",
                                       facecolor='#ECEFF1', edgecolor='#CFD8DC', lw=1.0, zorder=6)
    ax.add_patch(inner_box)

    # 3D Feature Map Slabs
    slabs = [
        (7.50, 4.90, 0.38, 2.1, '#3B82F6', '#93C5FD', '#1D4ED8', '#1E3A8A', 'Conv1\n64ch'),
        (8.20, 5.05, 0.42, 1.7, '#06B6D4', '#67E8F9', '#0E7490', '#155E75', 'Res2\n128ch'),
        (8.95, 5.20, 0.46, 1.4, '#6366F1', '#A5B4FC', '#4338CA', '#312E81', 'Res3\n256ch'),
        (9.75, 5.35, 0.50, 1.1, '#8B5CF6', '#C4B5FD', '#6D28D9', '#4C1D95', 'Res4\n512ch')
    ]
    for sx, sy, sw, sh, fc, tc, sc, ec, tag in slabs:
        draw_3d_block(ax, sx, sy, sw, sh, depth=0.26, angle=35,
                      face_col=fc, top_col=tc, side_col=sc, edge_col=ec, zorder=10)
        ax.text(sx + sw/2, sy - 0.26, tag, ha='center', va='top', color='#334155', fontsize=6.8, fontweight='bold', zorder=25)

    # Arrow from Res4 to Global AvgPool
    draw_arrow(ax, 10.40, 5.9, 10.70, 5.9, color='#64748B', lw=1.4, zorder=15)

    # Global AvgPool pill
    draw_pill(ax, 10.75, 5.35, 0.90, 1.10, text="Global\nAvgPool\n& Flatten",
              face_col='#FFFFFF', edge_col='#0284C7', text_col='#0369A1', font_size=7.0, font_weight='bold')

    draw_arrow(ax, 11.68, 5.9, 11.95, 5.9, color='#64748B', lw=1.4, zorder=15)

    # 768-D segmented bar
    draw_3d_block(ax, 12.00, 4.85, 0.28, 2.2, depth=0.20, angle=35,
                  face_col='#334155', top_col='#64748B', side_col='#0F172A', edge_col='#020617', zorder=10)
    for cy in np.linspace(5.0, 6.8, 8):
        ax.plot([12.00, 12.28], [cy, cy], color='#94A3B8', lw=0.8, zorder=15)
    ax.text(12.00, 4.55, "768-D Vector", ha='center', va='top', color='#0F172A', fontsize=7.2, fontweight='bold', zorder=25)

    # Downward flow to PCA feature selection with badge
    draw_arrow(ax, 9.95, 4.4, 9.95, 3.55, color='#0284C7', lw=1.8, badge='Train-Only Fit', badge_col='#0284C7', zorder=15)

    # Lower inner box: Train-Only PCA
    fs_box = patches.FancyBboxPatch((7.30, 1.15), 5.3, 2.3, boxstyle="round,pad=0.1,rounding_size=0.15",
                                    facecolor='#ECEFF1', edgecolor='#CFD8DC', lw=1.0, zorder=6)
    ax.add_patch(fs_box)
    
    # Clean Funnel Graphic on left (x = 7.45 to 8.05)
    funnel_pts = np.array([[7.45, 3.05], [8.05, 3.05], [7.83, 2.30], [7.67, 2.30]])
    funnel = patches.Polygon(funnel_pts, closed=True, facecolor='#E0F2FE', edgecolor='#0284C7', lw=1.8, zorder=15)
    ax.add_patch(funnel)
    ax.plot([7.75, 7.75], [2.30, 1.80], color='#0284C7', lw=2.0, zorder=15)
    ax.text(7.75, 1.55, "k = 5", ha='center', va='center', color='#0284C7', fontsize=8.0, fontweight='bold', zorder=25)

    # Explanatory text strictly on the right (x = 8.25 to 12.4, zero overlap with funnel!)
    ax.text(8.25, 2.95, "Strict Train-Only PCA Reduction", ha='left', va='center',
            color='#0F172A', fontsize=9.2, fontweight='bold', zorder=25)
    ax.text(8.25, 2.50, "- StandardScaler & PCA fit strictly on Train (n=180)\n- 5 Orthogonal Components explaining 63.2% Variance",
            ha='left', va='center', color='#334155', fontsize=7.8, zorder=25)
    ax.text(8.25, 1.85, "Z_mri in R^5  (Zero Validation / Pool Contamination)", ha='left', va='center',
            color='#0369A1', fontsize=7.8, fontweight='bold', zorder=25)

    draw_arrow(ax, 12.95, 4.8, 13.35, 4.8, color='#475569', lw=1.8, zorder=15)

    # 4. SUPERVISED CLASSIFIER PANEL (Right)
    panel_clf = patches.FancyBboxPatch((13.35, 0.8), 3.2, 8.0, boxstyle="round,pad=0.15,rounding_size=0.2",
                                       facecolor='#344955', edgecolor='#232F34', lw=1.4, zorder=5)
    ax.add_patch(panel_clf)
    ax.text(14.95, 8.32, "Cost-Sensitive Metric Head\n(Zero Data Leakage)", ha='center', va='center',
            color='white', fontsize=10.2, fontweight='bold', zorder=25)

    # Model pill
    draw_pill(ax, 13.55, 6.00, 2.8, 1.5,
              text="L2-Regularized\nLogistic Regression\n(C = 0.1, class_weight='balanced')",
              face_col='#ECEFF1', edge_col='#CFD8DC', text_col='#0F172A', font_size=8.0, font_weight='bold')
    
    draw_arrow(ax, 14.95, 6.00, 14.95, 5.10, color='#0284C7', lw=1.6, zorder=15)

    # Probability output card
    draw_pill(ax, 13.55, 4.00, 2.8, 1.1, text="Predicted Risk:\nP_MRI in [0, 1]",
              face_col='#E0F2FE', edge_col='#0284C7', text_col='#0369A1', font_size=9.0, font_weight='bold')

    draw_arrow(ax, 14.95, 4.00, 14.95, 3.05, color='#0284C7', lw=1.6, zorder=15)

    # Metrics dark pill
    draw_pill(ax, 13.55, 1.15, 2.8, 1.8,
              text="Held-Out Test Performance:\nTest ROC-AUC: 0.7221\nPR-AUC: 0.3185\nSensitivity: 71.4% (5/7)\nSpecificity: 38.2% (21/55)\nBrier Score: 0.2266",
              face_col='#0F172A', edge_col='#334155', text_col='#38BDF8', font_size=7.8, font_weight='bold')

    fig.savefig(OUT_DIR / "Fig2_MRI_Preprocessing.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / "Fig2_MRI_Preprocessing.tif", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Fig2_MRI_Preprocessing.png & .tif successfully generated.")


# =========================================================================
# FIGURE 3: MODULE 2 — BEHAVIORAL QUESTIONNAIRES PIPELINE
# =========================================================================
def build_fig3():
    print("Generating Figure 3: Module 2 Questionnaire Pipeline...")
    fig, ax = plt.subplots(figsize=(16.8, 9.0), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0.2, 9.2)
    ax.axis('off')

    # 1. INPUT 3D CARD (Left)
    draw_3d_block(ax, 0.4, 4.2, 2.2, 2.8, depth=0.28, angle=35,
                  face_col='#239B56', top_col='#58D68D', side_col='#196F3D', edge_col='#0E4B24', zorder=10)
    draw_clipboard_vector(ax, 1.50, 5.85, s=0.35, col='#E8F8F5', zorder=30)
    ax.text(1.50, 4.85, "109 Candidate Items\nSCAS, SPHERE,\nSDQ, MFQ", ha='center', va='center',
            color='white', fontsize=8.4, fontweight='bold', zorder=30)

    draw_arrow(ax, 2.7, 5.6, 3.2, 5.6, color='#239B56', lw=1.8, zorder=15)

    # 2. TRANSITION SELECTION CARD
    draw_pill(ax, 3.2, 3.3, 2.9, 4.6,
              text="Transition Selection\n(Zero Polarity Inversion)\n\n9 Clinically Guided Markers:\n- Somatic: pSDQ03,\n  SPHERE07, 08, 22\n- Social Fear: pSCAS31,\n  SCAS_socpho\n- Discrepancy:\n  diff_26, diff_worry\n\n(Breaks longitudinal\nsymptom inversion)",
              face_col='#FFFFFF', edge_col='#239B56', text_col='#0F172A', font_size=8.0, font_weight='normal')

    draw_arrow(ax, 6.2, 5.6, 6.7, 5.6, color='#239B56', lw=1.8, zorder=15)

    # 3. FEATURE-TOKENIZER ATTENTION BACKBONE (Large Center Panel)
    panel_ft = patches.FancyBboxPatch((6.7, 0.8), 4.8, 8.0, boxstyle="round,pad=0.15,rounding_size=0.2",
                                      facecolor='#FFFFFF', edgecolor='#239B56', lw=1.6, zorder=5)
    ax.add_patch(panel_ft)
    ax.text(9.10, 8.35, "Feature-Tokenizer Attention", ha='center', va='center',
            color='#239B56', fontsize=11.0, fontweight='bold', zorder=25)
    ax.text(9.10, 8.00, "Intra-Sample Attention (Batch Size 1 Invariant)", ha='center', va='center',
            color='#64748B', fontsize=8.2, fontstyle='italic', zorder=25)

    # Tokenizer 3D block
    draw_3d_block(ax, 7.1, 6.45, 4.0, 1.25, depth=0.22, angle=35,
                  face_col='#E0F2FE', top_col='#F0F9FF', side_col='#BAE6FD', edge_col='#0284C7', zorder=10)
    ax.text(9.10, 7.15, "Feature Tokenizer", ha='center', va='center',
            color='#0369A1', fontsize=9.5, fontweight='bold', zorder=30)
    ax.text(9.10, 6.78, "Projects 9 items -> 9 distinct 64-D continuous tokens", ha='center', va='center',
            color='#475569', fontsize=7.5, zorder=30)

    draw_arrow(ax, 9.10, 6.45, 9.10, 5.95, color='#239B56', lw=1.6, zorder=15)

    # Self-Attention 3D block
    draw_3d_block(ax, 7.1, 4.60, 4.0, 1.35, depth=0.22, angle=35,
                  face_col='#E8F8F5', top_col='#F4FBF9', side_col='#C1E8DF', edge_col='#239B56', zorder=10)
    ax.text(9.10, 5.40, "Self-Attention Blocks (x2)", ha='center', va='center',
            color='#196F3D', fontsize=9.5, fontweight='bold', zorder=30)
    ax.text(9.10, 4.95, "Multi-Head Attention (4 heads) + Residual + GELU MLP", ha='center', va='center',
            color='#475569', fontsize=7.5, zorder=30)

    draw_arrow(ax, 9.10, 4.60, 9.10, 4.10, color='#239B56', lw=1.6, zorder=15)

    # Attention Pooling 3D block (Roomy and well padded!)
    draw_3d_block(ax, 7.1, 2.70, 4.0, 1.45, depth=0.22, angle=35,
                  face_col='#FEF9E7', top_col='#FFFDF7', side_col='#F9E79F', edge_col='#D4AC0D', zorder=10)
    ax.text(9.10, 3.65, "Attention Pooling & Projection", ha='center', va='center',
            color='#B7950B', fontsize=9.5, fontweight='bold', zorder=30)
    ax.text(9.10, 3.28, "Attention-weighted token sum -> Linear 16-D Latent Space", ha='center', va='center',
            color='#475569', fontsize=7.6, zorder=30)

    # Sample independence note at bottom of center panel
    ax.text(9.10, 1.65, "Verified Sample-Independence:\nBatch vs Single-item diff = 2.50e-07\nZero Cross-Subject Leakage",
            ha='center', va='center', color='#196F3D', fontsize=7.8, fontweight='bold', zorder=25)

    draw_arrow(ax, 11.6, 5.0, 12.05, 5.0, color='#239B56', lw=1.8, zorder=15)

    # 4. METRIC PROTOTYPE SPACE (Biplot box)
    proto_box = patches.FancyBboxPatch((12.05, 2.2), 2.3, 5.5, boxstyle="round,pad=0.1,rounding_size=0.15",
                                       facecolor='#FFFFFF', edgecolor='#239B56', lw=1.3, zorder=6)
    ax.add_patch(proto_box)
    ax.text(13.20, 7.30, "Metric Prototype Space", ha='center', va='center',
            color='#239B56', fontsize=8.8, fontweight='bold', zorder=25)

    # 2D coordinate axes (well spaced, no label clipping)
    bx, by = 13.20, 5.15
    ax.plot([bx-0.85, bx+0.85], [by-0.3, by-0.3], color='#94A3B8', lw=1.2, zorder=15)
    ax.plot([bx-0.35, bx-0.35], [by-0.9, by+0.9], color='#94A3B8', lw=1.2, zorder=15)
    ax.text(bx+0.72, by-0.48, "z_1", ha='center', va='center', color='#64748B', fontsize=7.2, fontweight='bold', zorder=25)
    ax.text(bx-0.58, by+0.75, "z_2", ha='center', va='center', color='#64748B', fontsize=7.2, fontweight='bold', zorder=25)

    # Controls (green dots) and Incident cases (red dots)
    np.random.seed(42)
    pts_neg_x = np.random.normal(bx - 0.12, 0.15, 12)
    pts_neg_y = np.random.normal(by - 0.05, 0.14, 12)
    ax.scatter(pts_neg_x, pts_neg_y, s=26, color='#10B981', edgecolors='#047857', lw=0.8, zorder=25)

    pts_pos_x = np.random.normal(bx + 0.45, 0.14, 6)
    pts_pos_y = np.random.normal(by + 0.35, 0.14, 6)
    ax.scatter(pts_pos_x, pts_pos_y, s=30, color='#EF4444', edgecolors='#991B1B', lw=0.8, zorder=25)

    # Prototypes
    ax.scatter([bx - 0.12], [by - 0.05], marker='*', s=85, color='#047857', edgecolors='white', lw=1.0, zorder=30)
    ax.scatter([bx + 0.45], [by + 0.35], marker='*', s=85, color='#B91C1C', edgecolors='white', lw=1.0, zorder=30)

    ax.text(13.20, 3.35, "Green*: c_0 (Control)\nRed*: c_1 (Incident)", ha='center', va='center',
            color='#475569', fontsize=7.2, fontweight='bold', zorder=25)
    ax.text(13.20, 2.70, "d_k = ||z - c_k||_2\nP = exp(-d_1) / sum", ha='center', va='center',
            color='#196F3D', fontsize=7.2, fontstyle='italic', zorder=25)

    draw_arrow(ax, 14.45, 5.0, 14.85, 5.0, color='#239B56', lw=1.8, zorder=15)

    # 5. EPISODIC METRIC EVALUATION PANEL (Right)
    panel_eval = patches.FancyBboxPatch((14.85, 1.2), 1.6, 7.0, boxstyle="round,pad=0.15,rounding_size=0.2",
                                        facecolor='#196F3D', edgecolor='#0E4B24', lw=1.4, zorder=6)
    ax.add_patch(panel_eval)
    ax.text(15.65, 7.75, "Episodic Metric\nEvaluation", ha='center', va='center',
            color='white', fontsize=9.2, fontweight='bold', zorder=25)
    ax.text(15.65, 6.75, "- Euclidean dist\n- Temp tau = 1.0\n- Focal Loss", ha='center', va='center',
            color='#D1FAE5', fontsize=7.8, zorder=25)

    draw_arrow(ax, 15.65, 6.30, 15.65, 5.75, color='#A7F3D0', lw=1.4, zorder=15)

    # Output pill
    draw_pill(ax, 14.98, 4.80, 1.34, 0.90, text="P_Quest in [0, 1]",
              face_col='#FFFFFF', edge_col='#0E4B24', text_col='#196F3D', font_size=7.4, font_weight='bold')

    draw_arrow(ax, 15.65, 4.80, 15.65, 4.15, color='#A7F3D0', lw=1.4, zorder=15)

    # Metrics
    ax.text(15.65, 2.65, "Held-Out Metrics:\nVal AUC: 0.7455\nTest AUC: 0.5247\nSens: 57.1% (4/7)\nSpec: 67.3% (37/55)\nNPV: 92.5%",
            ha='center', va='center', color='#FEF08A', fontsize=7.5, fontweight='bold', zorder=25)

    fig.savefig(OUT_DIR / "Fig3_Questionnaire_Pipeline.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / "Fig3_Questionnaire_Pipeline.tif", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Fig3_Questionnaire_Pipeline.png & .tif successfully generated.")


# =========================================================================
# FIGURE 4: DETAILED DEEP PROTONET ARCHITECTURE (MAGNIFICENT & COMPLETE)
# =========================================================================
def build_fig4():
    print("Generating Figure 4: Detailed Deep Feature-Tokenizer ProtoNet Architecture...")
    fig, ax = plt.subplots(figsize=(17.2, 9.4), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0.2, 9.6)
    ax.axis('off')

    # 1. INPUT 3D PILLAR (Left)
    draw_3d_block(ax, 0.35, 2.8, 1.4, 4.4, depth=0.25, angle=35,
                  face_col='#E2E8F0', top_col='#F8FAFC', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(1.05, 5.30, "Input Vector x\nin R^D\n\n(D=9 for M2,\nD=4 for M3)",
            ha='center', va='center', color='#0F172A', fontsize=8.8, fontweight='bold', zorder=30)

    draw_arrow(ax, 1.90, 5.2, 2.30, 5.2, color='#64748B', lw=1.8, zorder=15)

    # 2. FEATURE TOKENIZER COLUMN
    panel_tok = patches.FancyBboxPatch((2.30, 0.8), 2.5, 8.2, boxstyle="round,pad=0.15,rounding_size=0.2",
                                       facecolor='#F0F9FF', edgecolor='#0284C7', lw=1.4, zorder=5)
    ax.add_patch(panel_tok)
    ax.text(3.55, 8.65, "Feature Tokenizer", ha='center', va='center',
            color='#0369A1', fontsize=10.5, fontweight='bold', zorder=25)
    ax.text(3.55, 8.28, "Independent Linear(1 -> 64)", ha='center', va='center',
            color='#0284C7', fontsize=7.5, fontstyle='italic', zorder=25)

    # 5 Stacks of 3D token slabs
    for idx, t_name in enumerate(["Token 1 in R^64", "Token 2 in R^64", "Token 3 in R^64", "Token 4 in R^64", "Token 5 in R^64"]):
        ty = 7.0 - idx * 1.15
        draw_3d_block(ax, 2.55, ty, 2.0, 0.65, depth=0.18, angle=35,
                      face_col='#FFFFFF', top_col='#F0F9FF', side_col='#BAE6FD', edge_col='#0284C7', zorder=10)
        ax.text(3.55, ty + 0.32, t_name, ha='center', va='center', color='#0369A1', fontsize=7.8, fontweight='bold', zorder=25)

    ax.text(3.55, 1.55, "Feature-Wise Embeddings\nMatrix T in R^(D x 64)", ha='center', va='center',
            color='#0369A1', fontsize=7.5, fontweight='bold', zorder=25)

    draw_arrow(ax, 4.95, 5.2, 5.35, 5.2, color='#0284C7', lw=1.8, zorder=15)

    # 3. MULTI-HEAD SELF-ATTENTION TRANSFORMER BLOCKS (Center Panel with Curved Residual Arrow!)
    panel_sa = patches.FancyBboxPatch((5.35, 0.8), 4.6, 8.2, boxstyle="round,pad=0.15,rounding_size=0.2",
                                      facecolor='#F0FDF4', edgecolor='#16A34A', lw=1.4, zorder=5)
    ax.add_patch(panel_sa)
    ax.text(7.65, 8.65, "Multi-Head Self-Attention Blocks (x2)", ha='center', va='center',
            color='#15803D', fontsize=10.5, fontweight='bold', zorder=25)

    # Multi-head attention inner pill
    draw_pill(ax, 5.75, 6.05, 3.9, 2.0,
              text="Multi-Head Attention (4 Heads)\nQ, K, V in R^(D x 64)\nAttn = softmax(Q K^T / sqrt(16))\nattn_out = Attn @ V",
              face_col='#FFFFFF', edge_col='#16A34A', text_col='#166534', font_size=8.2, font_weight='bold')

    # Curved Residual Connection Arrow looping cleanly INSIDE panel!
    ax.annotate('', xy=(5.75, 4.80), xytext=(5.75, 7.40),
                arrowprops=dict(arrowstyle="->", lw=1.6, color='#16A34A', connectionstyle="arc3,rad=-0.32"),
                zorder=20)
    ax.text(5.48, 6.10, "Residual (+)", ha='center', va='center', color='#16A34A',
            fontsize=7.2, fontweight='bold', rotation=90, zorder=25)

    draw_arrow(ax, 7.65, 6.05, 7.65, 5.35, color='#16A34A', lw=1.6, zorder=15)

    # LayerNorm pill
    draw_pill(ax, 5.75, 4.45, 3.9, 0.90,
              text="Residual Add (+) & LayerNorm(x + attn_out)",
              face_col='#DCFCE7', edge_col='#16A34A', text_col='#14532D', font_size=8.0, font_weight='bold')

    draw_arrow(ax, 7.65, 4.45, 7.65, 3.75, color='#16A34A', lw=1.6, zorder=15)

    # Position-wise Feedforward MLP pill
    draw_pill(ax, 5.75, 1.65, 3.9, 2.1,
              text="Position-wise Feed-Forward MLP\nLinear(64 -> 128) -> GELU -> Dropout(0.2)\nLinear(128 -> 64) + Residual\nLayerNorm(x + ffn_out)",
              face_col='#FFFFFF', edge_col='#16A34A', text_col='#166534', font_size=8.0, font_weight='bold')

    # Second curved residual loop for MLP
    ax.annotate('', xy=(5.75, 2.05), xytext=(5.75, 3.45),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#16A34A', connectionstyle="arc3,rad=-0.30"),
                zorder=20)

    draw_arrow(ax, 10.00, 5.2, 10.40, 5.2, color='#16A34A', lw=1.8, zorder=15)

    # 4. ATTENTION POOLING & PROJECTION
    panel_pool = patches.FancyBboxPatch((10.40, 0.8), 2.5, 8.2, boxstyle="round,pad=0.15,rounding_size=0.2",
                                        facecolor='#FFFBEB', edgecolor='#D97706', lw=1.4, zorder=5)
    ax.add_patch(panel_pool)
    ax.text(11.65, 8.65, "Attention Pooling", ha='center', va='center',
            color='#B45309', fontsize=10.5, fontweight='bold', zorder=25)

    draw_pill(ax, 10.60, 5.15, 2.1, 2.6,
              text="Learnable\nAttention Weights\n\nalpha_i in [0, 1]\nsum alpha_i = 1\n\nh = sum alpha_i * t_i\nh in R^64",
              face_col='#FFFFFF', edge_col='#D97706', text_col='#92400E', font_size=8.0, font_weight='bold')

    draw_arrow(ax, 11.65, 5.15, 11.65, 4.25, color='#D97706', lw=1.6, zorder=15)

    draw_pill(ax, 10.60, 1.85, 2.1, 2.4,
              text="Projection MLP\n\nLinear(64 -> 32)\nLinear(32 -> 16)\n\nz in R^16\n(Latent Embedding)",
              face_col='#FEF3C7', edge_col='#D97706', text_col='#78350F', font_size=8.0, font_weight='bold')

    draw_arrow(ax, 13.05, 5.2, 13.45, 5.2, color='#D97706', lw=1.8, zorder=15)

    # 5. EPISODIC FEW-SHOT METRIC HEAD (Right Panel with Realistic 3D Spheres!)
    panel_head = patches.FancyBboxPatch((13.45, 0.8), 3.4, 8.2, boxstyle="round,pad=0.15,rounding_size=0.2",
                                        facecolor='#FAF5FF', edgecolor='#9333EA', lw=1.6, zorder=5)
    ax.add_patch(panel_head)
    ax.text(15.15, 8.65, "Episodic Few-Shot Metric Head", ha='center', va='center',
            color='#7E22CE', fontsize=10.2, fontweight='bold', zorder=25)

    # Negative Prototype 3D Sphere & Pill (Green)
    draw_sphere(ax, 14.05, 7.55, r=0.28, base_col='#10B981', highlight_col='#6EE7B7', edge_col='#047857', zorder=30)
    draw_pill(ax, 14.45, 7.25, 2.2, 0.65, text="c_0: Control Prototype",
              face_col='#DCFCE7', edge_col='#16A34A', text_col='#15803D', font_size=7.6, font_weight='bold')

    # Positive Prototype 3D Sphere & Pill (Red)
    draw_sphere(ax, 14.05, 6.45, r=0.28, base_col='#EF4444', highlight_col='#FCA5A5', edge_col='#B91C1C', zorder=30)
    draw_pill(ax, 14.45, 6.15, 2.2, 0.65, text="c_1: Incident Prototype",
              face_col='#FEE2E2', edge_col='#DC2626', text_col='#B91C1C', font_size=7.6, font_weight='bold')

    # Convergence arrows from prototypes to metric computation
    draw_arrow(ax, 15.15, 6.15, 15.15, 5.45, color='#9333EA', lw=1.4, zorder=15)

    # Euclidean distance metric pill (Spacious, zero border clipping!)
    draw_pill(ax, 13.65, 3.55, 3.0, 1.9,
              text="Euclidean Distance Metric:\nd_k = || z - c_k ||_2\n\nP(y=1|x) = exp(-d_1) / sum exp(-d_k)\n(Strict Sample Independence)",
              face_col='#FFFFFF', edge_col='#9333EA', text_col='#6B21A8', font_size=8.0, font_weight='bold')

    draw_arrow(ax, 15.15, 3.55, 15.15, 2.75, color='#9333EA', lw=1.6, zorder=15)

    # Focal loss pill
    draw_pill(ax, 13.65, 1.15, 3.0, 1.6,
              text="Optimization Objective:\nClass-Weighted Focal Loss\n(gamma = 2.0, alpha = 0.75)\nHandles ~1:6.5 Class Imbalance",
              face_col='#7E22CE', edge_col='#581C87', text_col='#F3E8FF', font_size=8.0, font_weight='bold')

    fig.savefig(OUT_DIR / "Fig4_Prototypical_Network.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / "Fig4_Prototypical_Network.tif", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Fig4_Prototypical_Network.png & .tif successfully generated.")


if __name__ == "__main__":
    print("=" * 70)
    print("REGENERATING ALL 4 PUBLICATION ARCHITECTURE FIGURES (300 DPI)")
    print("=" * 70)
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    print("=" * 70)
    print("ALL 4 FIGURES SUCCESSFULLY REGENERATED IN results/")
    print("=" * 70)
