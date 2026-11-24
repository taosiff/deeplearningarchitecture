"""
code/generate_publication_architectures.py
=========================================
Publication-Grade 3D Scientific Architecture Figures (Figs 1, 2, 3, 4)
Logical, Mathematical, and Aesthetic Perfection for Q1 Journal Review

Generates:
  - Fig 1: Overall Multimodal Framework (Parameter-Free Simple Average Decision Fusion)
  - Fig 2: Module 1 — Structural Neuroimaging Pipeline (Med3D 3D-ResNet-18 + Train-Only PCA-5 + Balanced LR)
  - Fig 3: Module 2 — Longitudinal Anxiety Symptomatology Pipeline (3D Isometric Glass Box + APN + Prototypes Cube + Multi-Loss Feedback)
  - Fig 4: Detailed Deep Prototypical Network Architecture (6-Column Isometric Architecture with Dual Encoders, Residual Loops, Prototype Spheres)

Aesthetic & Technical Standards:
  - Pure white background (#FFFFFF), high contrast, crisp vectors
  - Clean isometric 3D beveled blocks with top highlight and right shading
  - Translucent 3D wireframe boxes and isometric prototype coordinate hexagons
  - Curved 3D residual skip loops and 3D shaded prototype spheres with radial specular highlights
  - Zero text-icon collisions, comfortable margins, and mathematically honest metrics
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

def draw_translucent_cube(ax, x, y, w, h, depth=0.6, angle=30,
                          face_col='#E0F2FE', edge_col='#0284C7', alpha=0.30, lw=1.4, zorder=10):
    """Draws a translucent 3D isometric glass cuboid."""
    rad = np.radians(angle)
    dx = depth * np.cos(rad)
    dy = depth * np.sin(rad)
    
    # Back wireframe edges
    ax.plot([x + dx, x + w + dx], [y + dy, y + dy], color=edge_col, lw=1.1, linestyle=':', zorder=zorder)
    ax.plot([x + dx, x + dx], [y + dy, y + h + dy], color=edge_col, lw=1.1, linestyle=':', zorder=zorder)
    ax.plot([x, x + dx], [y, y + dy], color=edge_col, lw=1.1, linestyle=':', zorder=zorder)

    # Front face
    front = patches.Rectangle((x, y), w, h, facecolor=face_col,
                              edgecolor=edge_col, alpha=alpha, linewidth=lw, zorder=zorder+2)
    ax.add_patch(front)

    # Top face
    top = patches.Polygon([[x, y+h], [x+w, y+h], [x+w+dx, y+h+dy], [x+dx, y+h+dy]], closed=True,
                          facecolor=face_col, edgecolor=edge_col, alpha=alpha+0.15, linewidth=lw, zorder=zorder+3)
    ax.add_patch(top)

    # Right side face
    side = patches.Polygon([[x+w, y], [x+w+dx, y+dy], [x+w+dx, y+h+dy], [x+w, y+h]], closed=True,
                           facecolor=face_col, edgecolor=edge_col, alpha=alpha+0.10, linewidth=lw, zorder=zorder+3)
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
    ax.add_patch(patches.Circle((cx, cy), r, facecolor=base_col, edgecolor=edge_col, lw=1.2, zorder=zorder))
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

def draw_clipboard_vector(ax, cx, cy, s=0.28, col='#0284C7', zorder=35):
    """Draws a crisp clinical clipboard vector with checklist."""
    w, h = s * 0.90, s * 1.25
    ax.add_patch(patches.Rectangle((cx - w/2, cy - h/2), w, h, facecolor='white', edgecolor=col, lw=1.4, zorder=zorder))
    ax.add_patch(patches.Rectangle((cx - w*0.3, cy + h*0.38), w*0.6, h*0.18, facecolor=col, edgecolor=col, lw=0.8, zorder=zorder+1))
    for dy in [-0.25, 0.05, 0.32]:
        ax.plot([cx - w*0.12, cx + w*0.32], [cy + dy*h*0.65, cy + dy*h*0.65], color='#64748B', lw=1.2, zorder=zorder+1)
        ax.plot([cx - w*0.35, cx - w*0.25, cx - w*0.18], [cy + dy*h*0.65, cy + dy*h*0.65 - 0.04, cy + dy*h*0.65 + 0.05],
                color='#10B981', lw=1.3, zorder=zorder+1)

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
    draw_clipboard_vector(ax, 6.80, 8.02, s=0.25, col='#FFFFFF', zorder=30)
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

    # Clean diagonal arrows with prominent w = 1/3 weight badges
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
    
    # Clean Funnel Graphic on left
    funnel_pts = np.array([[7.45, 3.05], [8.05, 3.05], [7.83, 2.30], [7.67, 2.30]])
    funnel = patches.Polygon(funnel_pts, closed=True, facecolor='#E0F2FE', edgecolor='#0284C7', lw=1.8, zorder=15)
    ax.add_patch(funnel)
    ax.plot([7.75, 7.75], [2.30, 1.80], color='#0284C7', lw=2.0, zorder=15)
    ax.text(7.75, 1.55, "k = 5", ha='center', va='center', color='#0284C7', fontsize=8.0, fontweight='bold', zorder=25)

    # Explanatory text strictly on the right
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
# (Faithful Recreation of Previous Submission Fig 3: 3D Glass Box + APN + Prototypes Cube + Feedback)
# =========================================================================
def build_fig3():
    print("Generating Figure 3: Module 2 Questionnaire Pipeline (3D Isometric Masterpiece)...")
    fig, ax = plt.subplots(figsize=(16.8, 8.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0.2, 8.4)
    ax.axis('off')

    # Top Performance Banner (Placed cleanly above Preprocessing & Augmentation, zero overlap with APN or Prototypes!)
    draw_pill(ax, 3.2, 7.35, 5.0, 0.48,
              text="Held-Out Test: ROC-AUC = 0.5247 | Sens = 57.1% | Spec = 67.3% | NPV = 92.5%",
              face_col='#F0FDF4', edge_col='#16A34A', text_col='#15803D', font_size=7.4, font_weight='bold')

    # 1. INPUT: 3D TRANSLUCENT GLASS ISOMETRIC BOX (Left)
    draw_translucent_cube(ax, 0.45, 3.2, 1.85, 2.9, depth=0.65, angle=32,
                          face_col='#E0F2FE', edge_col='#0284C7', alpha=0.32, lw=1.5, zorder=10)
    # Suspended clipboard inside glass cube
    draw_clipboard_vector(ax, 1.38, 4.65, s=0.42, col='#0284C7', zorder=20)
    ax.text(1.38, 2.65, "Input", ha='center', va='center',
            color='#0F172A', fontsize=12.0, fontweight='bold', zorder=30)
    ax.text(1.38, 2.25, "(109 Baseline Items)", ha='center', va='center',
            color='#64748B', fontsize=8.2, fontstyle='italic', zorder=30)

    # Thick cyan arrow to Preprocessing
    draw_arrow(ax, 2.75, 4.65, 3.25, 4.65, color='#00A3C4', lw=2.6, zorder=15)

    # 2. PREPROCESSING (Rounded Cyan Card)
    box_prep = patches.FancyBboxPatch((3.25, 3.80), 2.05, 1.70, boxstyle="round,pad=0.1,rounding_size=0.18",
                                      facecolor='#FFFFFF', edgecolor='#00A3C4', lw=1.5, zorder=12)
    ax.add_patch(box_prep)
    ax.text(4.27, 4.78, "Preprocessing", ha='center', va='center',
            color='#0F172A', fontsize=10.5, fontweight='bold', zorder=25)
    ax.text(4.27, 4.25, "In-Fold Imputation\n& Robust Scaling", ha='center', va='center',
            color='#0284C7', fontsize=7.8, zorder=25)

    # Thick cyan arrow to Augmentation / Transition Selection
    draw_arrow(ax, 5.45, 4.65, 6.00, 4.65, color='#00A3C4', lw=2.6, zorder=15)

    # 3. TRANSITION SELECTION / AUGMENTATION (Rounded Cyan Card)
    box_aug = patches.FancyBboxPatch((6.00, 3.80), 2.15, 1.70, boxstyle="round,pad=0.1,rounding_size=0.18",
                                     facecolor='#FFFFFF', edgecolor='#00A3C4', lw=1.5, zorder=12)
    ax.add_patch(box_aug)
    ax.text(7.07, 4.78, "Augmentation", ha='center', va='center',
            color='#0F172A', fontsize=10.5, fontweight='bold', zorder=25)
    ax.text(7.07, 4.25, "Transition Markers\n(9 Clinical Items)", ha='center', va='center',
            color='#0284C7', fontsize=7.8, zorder=25)

    # Thick cyan arrow to Advanced Prototypical Network
    draw_arrow(ax, 8.30, 4.65, 8.85, 4.65, color='#00A3C4', lw=2.8, zorder=15)

    # 4. ADVANCED PROTOTYPICAL NETWORK (Large 3D Box)
    draw_3d_block(ax, 8.85, 2.70, 4.15, 4.40, depth=0.35, angle=35,
                  face_col='#FFFFFF', top_col='#F8FAFC', side_col='#E2E8F0', edge_col='#2C1337', lw=1.5, zorder=10)
    ax.text(10.92, 6.75, "Advanced Prototypical Network", ha='center', va='center',
            color='#0F172A', fontsize=11.2, fontweight='bold', zorder=25)

    # Downward arrow to Linear->LayerNorm->GELU->Dropout
    draw_arrow(ax, 10.92, 6.50, 10.92, 6.05, color='#4A235A', lw=1.6, zorder=20)

    # 3D Slab: Linear -> LayerNorm -> GELU -> Dropout
    draw_3d_block(ax, 9.15, 5.30, 3.55, 0.70, depth=0.20, angle=35,
                  face_col='#EEF2FF', top_col='#F5F3FF', side_col='#DDD6FE', edge_col='#6366F1', lw=1.0, zorder=15)
    ax.text(10.92, 5.65, "Linear -> LayerNorm -> GELU -> Dropout", ha='center', va='center',
            color='#4338CA', fontsize=8.0, fontweight='bold', zorder=25)

    # Downward arrow to Self-Attention + Residual
    draw_arrow(ax, 10.92, 5.30, 10.92, 4.65, color='#4A235A', lw=1.6, zorder=20)

    # 3D Block: Self-Attention + Residual
    draw_3d_block(ax, 9.45, 3.90, 2.95, 0.70, depth=0.20, angle=35,
                  face_col='#FFFFFF', top_col='#FAF5FF', side_col='#F3E8FF', edge_col='#2C1337', lw=1.2, zorder=15)
    ax.text(10.92, 4.25, "Self-Attention + Residual", ha='center', va='center',
            color='#0F172A', fontsize=8.2, fontweight='bold', zorder=25)

    # Circular Recurrence Loop Arrow (Smooth circuit looping back into bottom!)
    ax.plot([12.55, 12.80, 12.80], [4.25, 4.25, 3.30], color='#4A235A', lw=2.2, zorder=20)
    ax.plot([12.80, 11.25], [3.30, 3.30], color='#4A235A', lw=2.2, zorder=20)
    # Circular arc arrow tip
    arc = patches.Arc((10.90, 3.30), 0.70, 0.70, angle=0, theta1=40, theta2=320, color='#4A235A', lw=2.2, zorder=20)
    ax.add_patch(arc)
    ax.annotate('', xy=(10.90 + 0.35*np.cos(np.radians(40)), 3.30 + 0.35*np.sin(np.radians(40))),
                xytext=(10.90 + 0.35*np.cos(np.radians(35)), 3.30 + 0.35*np.sin(np.radians(35))),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color='#4A235A', mutation_scale=14),
                zorder=22)

    # Thick purple arrow to Prototypes Cube
    draw_arrow(ax, 13.15, 4.90, 13.75, 4.90, color='#4A235A', lw=3.2, zorder=25)

    # 5. PROTOTYPES: 3D ISOMETRIC HEXAGON / WIREFRAME CUBE (Right)
    cx, cy, R = 15.10, 4.90, 1.85
    rad30 = np.radians(30)
    hdx = R * np.cos(rad30)
    hdy = R * np.sin(rad30)
    pts_hex = np.array([
        [cx, cy + R],
        [cx + hdx, cy + hdy],
        [cx + hdx, cy - R + hdy],
        [cx, cy - R],
        [cx - hdx, cy - R + hdy],
        [cx - hdx, cy + hdy]
    ])
    hex_patch = patches.Polygon(pts_hex, closed=True, facecolor='#FFFFFF', edgecolor='#2C1337', lw=1.6, zorder=12)
    ax.add_patch(hex_patch)

    # 3 Internal coordinate axes (Matching prev_submissions isometric cube!)
    ax.plot([cx, cx], [cy, cy + R], color='#64748B', lw=1.1, linestyle='--', zorder=14)
    ax.plot([cx, cx], [cy, cy - R], color='#64748B', lw=1.1, linestyle='-', zorder=14)
    ax.plot([cx, cx + hdx], [cy, cy + hdy], color='#64748B', lw=1.1, linestyle='-', zorder=14)
    ax.plot([cx, cx - hdx], [cy, cy + hdy], color='#64748B', lw=1.1, linestyle='-', zorder=14)

    ax.text(cx, cy + R + 0.28, "Prototypes", ha='center', va='center',
            color='#0F172A', fontsize=11.5, fontweight='bold', zorder=25)

    # Class 0: Green cluster (Negative / Control)
    np.random.seed(42)
    c0_x = np.random.normal(cx - 0.70, 0.16, 6)
    c0_y = np.random.normal(cy + 0.15, 0.22, 6)
    ax.scatter(c0_x, c0_y, s=34, color='#10B981', edgecolors='#047857', lw=0.9, zorder=20)
    ax.scatter([cx - 0.70], [cy + 0.15], marker='*', s=140, color='#047857', edgecolors='white', lw=1.1, zorder=25)

    # Class 1: Red cluster (Positive / Incident Anxiety)
    c1_x = np.random.normal(cx + 0.65, 0.16, 6)
    c1_y = np.random.normal(cy + 0.45, 0.22, 6)
    ax.scatter(c1_x, c1_y, s=36, color='#EF4444', edgecolors='#991B1B', lw=0.9, zorder=20)
    ax.scatter([cx + 0.65], [cy + 0.45], marker='*', s=140, color='#B91C1C', edgecolors='white', lw=1.1, zorder=25)

    # Legend inside the prototype cube
    ax.scatter([cx - 0.60], [cy - 1.15], s=28, color='#10B981', edgecolors='#047857', lw=0.8, zorder=25)
    ax.text(cx - 0.40, cy - 1.15, "Class 0 (Neg)", ha='left', va='center', color='#0F172A', fontsize=7.6, fontweight='bold', zorder=25)
    ax.scatter([cx - 0.60], [cy - 1.45], s=28, color='#EF4444', edgecolors='#991B1B', lw=0.8, zorder=25)
    ax.text(cx - 0.40, cy - 1.45, "Class 1 (Pos)", ha='left', va='center', color='#0F172A', fontsize=7.6, fontweight='bold', zorder=25)

    # 6. BOTTOM FLOW & MULTI-LOSS FEEDBACK
    # Downward arrow from prototype cube to Inversion Check
    draw_arrow(ax, cx, cy - R, cx, 1.85, color='#93C5FD', lw=2.6, zorder=15)

    # Inversion Check 3D Block
    draw_3d_block(ax, cx - 1.05, 0.95, 2.10, 0.85, depth=0.20, angle=35,
                  face_col='#FFFFFF', top_col='#F8FAFC', side_col='#E2E8F0', edge_col='#93C5FD', lw=1.2, zorder=15)
    ax.text(cx, 1.38, "Inversion Check", ha='center', va='center',
            color='#0F172A', fontsize=8.8, fontweight='bold', zorder=25)

    # Feedback line flowing left across the bottom
    ax.plot([cx, cx], [0.95, 0.52], color='#93C5FD', lw=2.2, zorder=15)
    ax.plot([cx, 9.55], [0.52, 0.52], color='#93C5FD', lw=2.2, zorder=15)

    # 3 Loss Blocks with upward arrows into APN
    losses = [
        ("Contrastive\nLoss", 12.35),
        ("Center\nLoss", 10.95),
        ("Focal\nLoss", 9.55)
    ]
    for l_name, lx in losses:
        draw_arrow(ax, lx, 0.52, lx, 0.95, color='#93C5FD', lw=1.8, zorder=15)
        draw_3d_block(ax, lx - 0.65, 0.95, 1.30, 0.85, depth=0.18, angle=35,
                      face_col='#FFFFFF', top_col='#F8FAFC', side_col='#E2E8F0', edge_col='#93C5FD', lw=1.2, zorder=15)
        ax.text(lx, 1.38, l_name, ha='center', va='center', color='#0F172A', fontsize=7.6, fontweight='bold', zorder=25)
        draw_arrow(ax, lx, 1.80, lx, 2.70, color='#93C5FD', lw=1.8, zorder=15)

    fig.savefig(OUT_DIR / "Fig3_Questionnaire_Pipeline.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / "Fig3_Questionnaire_Pipeline.tif", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Fig3_Questionnaire_Pipeline.png & .tif successfully generated.")


# =========================================================================
# FIGURE 4: DETAILED DEEP PROTONET ARCHITECTURE
# (Faithful Recreation of Previous Submission Fig 4: 6-Column 3D Architecture)
# =========================================================================
def build_fig4():
    print("Generating Figure 4: Detailed Deep Feature-Tokenizer ProtoNet Architecture (Full 6-Column Grandeur)...")
    fig, ax = plt.subplots(figsize=(18.2, 9.6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0.2, 9.8)
    ax.axis('off')

    # Main Title (Centered at Top)
    ax.text(9.10, 9.40, "Advanced Prototypical Network Architecture", ha='center', va='center',
            color='#0F172A', fontsize=14.5, fontweight='bold', zorder=30)

    # -------------------------------------------------------------------------
    # COLUMN 1: STATIC FEATURES INPUT (3D Silver Pillar)
    # -------------------------------------------------------------------------
    draw_3d_block(ax, 0.45, 2.40, 0.75, 5.20, depth=0.25, angle=35,
                  face_col='#E2E8F0', top_col='#F8FAFC', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(0.82, 5.00, "Static Features Input (D-dim)", ha='center', va='center',
            color='#0F172A', fontsize=9.2, fontweight='bold', rotation=90, zorder=30)

    draw_arrow(ax, 1.35, 5.00, 1.75, 5.00, color='#0F172A', lw=1.8, zorder=15)

    # -------------------------------------------------------------------------
    # COLUMN 2: INPUT PROJECTION (3D Steel Blue Pillar)
    # -------------------------------------------------------------------------
    draw_3d_block(ax, 1.75, 2.15, 0.90, 5.70, depth=0.25, angle=35,
                  face_col='#4A90E2', top_col='#7EB2EC', side_col='#357ABD', edge_col='#1D4ED8', zorder=10)
    ax.text(2.20, 5.00, "Input Projection (Linear D -> 64)", ha='center', va='center',
            color='white', fontsize=9.5, fontweight='bold', rotation=90, zorder=30)

    # Circuit branching routing from Input Projection to Dual Encoders
    ax.plot([2.85, 3.10], [5.00, 5.00], color='#0F172A', lw=1.6, zorder=15)
    ax.plot([3.10, 3.10], [2.50, 7.00], color='#0F172A', lw=1.6, zorder=15)
    draw_arrow(ax, 3.10, 7.00, 3.30, 7.00, color='#0F172A', lw=1.6, zorder=15)
    draw_arrow(ax, 3.10, 2.50, 3.30, 2.50, color='#0F172A', lw=1.6, zorder=15)

    # -------------------------------------------------------------------------
    # COLUMN 3: STACKED DUAL ENCODER BLOCKS
    # -------------------------------------------------------------------------
    # Block 1 (Top)
    draw_3d_block(ax, 3.30, 5.20, 3.10, 3.65, depth=0.28, angle=35,
                  face_col='#6BA4D9', top_col='#91BFEF', side_col='#4682B4', edge_col='#1D4ED8', lw=1.2, zorder=10)
    ax.text(4.85, 8.52, "Dual Encoder\nBlock 1", ha='center', va='center',
            color='#0F172A', fontsize=9.8, fontweight='bold', zorder=25)

    slabs_b1 = [("Linear(64)", 7.75), ("LayerNorm", 7.15), ("GELU", 6.55), ("Dropout(0.2)", 5.95)]
    for s_name, sy in slabs_b1:
        draw_3d_block(ax, 3.55, sy, 2.15, 0.48, depth=0.18, angle=35,
                      face_col='#FFFFFF', top_col='#F8FAFC', side_col='#E2E8F0', edge_col='#1E293B', lw=1.0, zorder=15)
        ax.text(4.62, sy + 0.24, s_name, ha='center', va='center', color='#0F172A', fontsize=8.0, fontweight='bold', zorder=25)

    # Curved 3D black residual skip arrow for Block 1 (curving back inward into bottom!)
    ax.annotate('', xy=(5.70, 5.60), xytext=(5.70, 7.95),
                arrowprops=dict(arrowstyle="->", lw=1.8, color='black', connectionstyle="arc3,rad=-0.50"),
                zorder=22)
    ax.text(4.85, 5.48, "Residual\nConnection (+)", ha='center', va='center', color='#0F172A', fontsize=7.6, fontweight='bold', zorder=25)

    # Block 2 (Bottom)
    draw_3d_block(ax, 3.30, 0.65, 3.10, 3.65, depth=0.28, angle=35,
                  face_col='#6BA4D9', top_col='#91BFEF', side_col='#4682B4', edge_col='#1D4ED8', lw=1.2, zorder=10)
    ax.text(4.85, 3.97, "Dual Encoder\nBlock 2", ha='center', va='center',
            color='#0F172A', fontsize=9.8, fontweight='bold', zorder=25)

    slabs_b2 = [("Linear(64)", 3.20), ("LayerNorm", 2.60), ("GELU", 2.00), ("Dropout(0.2)", 1.40)]
    for s_name, sy in slabs_b2:
        draw_3d_block(ax, 3.55, sy, 2.15, 0.48, depth=0.18, angle=35,
                      face_col='#FFFFFF', top_col='#F8FAFC', side_col='#E2E8F0', edge_col='#1E293B', lw=1.0, zorder=15)
        ax.text(4.62, sy + 0.24, s_name, ha='center', va='center', color='#0F172A', fontsize=8.0, fontweight='bold', zorder=25)

    # Curved 3D black residual skip arrow for Block 2
    ax.annotate('', xy=(5.70, 1.05), xytext=(5.70, 3.40),
                arrowprops=dict(arrowstyle="->", lw=1.8, color='black', connectionstyle="arc3,rad=-0.50"),
                zorder=22)
    ax.text(4.85, 0.93, "Residual\nConnection (+)", ha='center', va='center', color='#0F172A', fontsize=7.6, fontweight='bold', zorder=25)

    # Merged circuit routing from Dual Encoders to Self-Attention
    ax.plot([6.55, 6.90], [7.00, 7.00], color='#0F172A', lw=1.6, zorder=15)
    ax.plot([6.55, 6.90], [2.50, 2.50], color='#0F172A', lw=1.6, zorder=15)
    ax.plot([6.90, 6.90], [2.50, 7.00], color='#0F172A', lw=1.6, zorder=15)
    draw_arrow(ax, 6.90, 7.00, 7.25, 7.00, color='#0F172A', lw=1.8, zorder=15)

    # -------------------------------------------------------------------------
    # COLUMN 4: SELF-ATTENTION & EMBEDDING PROJECTION
    # -------------------------------------------------------------------------
    # Self-Attention Mechanism (Sage Green 3D Block)
    draw_3d_block(ax, 7.25, 5.35, 3.30, 3.25, depth=0.28, angle=35,
                  face_col='#90C7B4', top_col='#B9DFC5', side_col='#6FA894', edge_col='#1E3A2F', lw=1.2, zorder=10)
    ax.text(8.90, 8.28, "Self-Attention\nMechanism", ha='center', va='center',
            color='#0F172A', fontsize=9.8, fontweight='bold', zorder=25)

    attn_slabs = [
        ("(Q, K, V Projections)", 7.35),
        ("Softmax(QK^T / sqrt(16))", 6.65),
        ("Residual Add & Norm", 5.95)
    ]
    for a_name, ay in attn_slabs:
        draw_3d_block(ax, 7.45, ay, 2.70, 0.52, depth=0.18, angle=35,
                      face_col='#FFFFFF', top_col='#F8FAFC', side_col='#E2E8F0', edge_col='#1E293B', lw=1.0, zorder=15)
        ax.text(8.80, ay + 0.26, a_name, ha='center', va='center', color='#0F172A', fontsize=7.8, fontweight='bold', zorder=25)

    # Downward arrow to Embedding Projection
    draw_arrow(ax, 8.90, 5.35, 8.90, 4.45, color='#0F172A', lw=1.6, zorder=20)

    # Embedding Projection (Linear 64 -> 16)
    draw_3d_block(ax, 7.25, 3.25, 3.30, 1.20, depth=0.25, angle=35,
                  face_col='#6BA4D9', top_col='#91BFEF', side_col='#4682B4', edge_col='#1D4ED8', lw=1.2, zorder=10)
    ax.text(8.90, 3.85, "Embedding\nProjection", ha='center', va='center', color='#0F172A', fontsize=9.5, fontweight='bold', zorder=25)
    ax.text(8.90, 3.45, "(Linear 64 -> 16)", ha='center', va='center', color='#1E293B', fontsize=7.8, zorder=25)

    # Downward arrow to L2 Normalization
    draw_arrow(ax, 8.90, 3.25, 8.90, 2.35, color='#0F172A', lw=1.6, zorder=20)

    # L2 Normalization (Unit Hypersphere)
    draw_3d_block(ax, 7.25, 1.15, 3.30, 1.20, depth=0.25, angle=35,
                  face_col='#E2E8F0', top_col='#F8FAFC', side_col='#CBD5E1', edge_col='#94A3B8', lw=1.2, zorder=10)
    ax.text(8.90, 1.75, "L2 Normalization", ha='center', va='center', color='#0F172A', fontsize=9.5, fontweight='bold', zorder=25)
    ax.text(8.90, 1.35, "(Unit Hypersphere)", ha='center', va='center', color='#475569', fontsize=7.8, zorder=25)

    # Arrow from Self-Attention mechanism to Multi-Head Prototype Learning
    draw_arrow(ax, 10.70, 6.00, 11.45, 6.00, color='#0F172A', lw=1.8, badge='16-dim', badge_col='#475569', zorder=20)

    # -------------------------------------------------------------------------
    # COLUMN 5: MULTI-HEAD PROTOTYPE LEARNING CONTAINER
    # -------------------------------------------------------------------------
    draw_3d_block(ax, 11.45, 0.65, 5.50, 8.25, depth=0.35, angle=35,
                  face_col='#F1F5F9', top_col='#FFFFFF', side_col='#CBD5E1', edge_col='#334155', lw=1.4, zorder=10)
    ax.text(14.20, 8.55, "Multi-Head Prototype Learning", ha='center', va='center',
            color='#0F172A', fontsize=11.2, fontweight='bold', zorder=25)

    # 4 Head Pills on left of panel
    head_ys = [7.20, 6.00, 4.80, 3.60]
    for idx, hy in enumerate(head_ys):
        draw_3d_block(ax, 11.75, hy, 1.45, 0.70, depth=0.18, angle=35,
                      face_col='#E0F2FE', top_col='#F0F9FF', side_col='#BAE6FD', edge_col='#0284C7', lw=1.0, zorder=15)
        ax.text(12.47, hy + 0.35, f"Head {idx+1}\n(4-dim)", ha='center', va='center',
                color='#0369A1', fontsize=7.8, fontweight='bold', zorder=25)

    # Top Shaded Prototype Sphere: Negative Prototype (Class 0)
    draw_sphere(ax, 14.70, 7.55, r=0.28, base_col='#3B82F6', highlight_col='#93C5FD', edge_col='#1D4ED8', zorder=25)
    ax.text(15.55, 7.72, "Negative\nPrototype\n(Class 0)", ha='left', va='center',
            color='#0F172A', fontsize=8.0, fontweight='bold', zorder=25)

    # Similarity Aggregation Block 0
    draw_3d_block(ax, 13.95, 5.55, 2.05, 1.15, depth=0.18, angle=35,
                  face_col='#DBEAFE', top_col='#EFF6FF', side_col='#BFDBFE', edge_col='#2563EB', lw=1.1, zorder=15)
    ax.text(14.97, 6.22, "Aggregate\nSimilarities\nto Class 0", ha='center', va='center',
            color='#1E40AF', fontsize=7.8, fontweight='bold', zorder=25)
    ax.text(14.97, 5.75, "S_0 = mean(S^h_c0 / tau)", ha='center', va='center',
            color='#1E3A8A', fontsize=6.5, zorder=25)

    # Lines from Head pills and Prototype 0 to Aggregation 0
    for hy in head_ys:
        ax.plot([13.35, 13.95], [hy + 0.35, 6.12], color='#0F172A', lw=1.0, zorder=20)
    ax.annotate('', xy=(14.70, 6.70), xytext=(14.70, 7.27),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color='#0F172A', mutation_scale=12), zorder=22)

    # Bottom Shaded Prototype Sphere: Positive Prototype (Class 1)
    draw_sphere(ax, 14.70, 1.95, r=0.28, base_col='#EF4444', highlight_col='#FCA5A5', edge_col='#B91C1C', zorder=25)
    ax.text(15.55, 1.95, "Positive\nPrototype\n(Class 1)", ha='left', va='center',
            color='#0F172A', fontsize=8.0, fontweight='bold', zorder=25)

    # Similarity Aggregation Block 1
    draw_3d_block(ax, 13.95, 3.55, 2.05, 1.15, depth=0.18, angle=35,
                  face_col='#FEE2E2', top_col='#FEF2F2', side_col='#FECACA', edge_col='#DC2626', lw=1.1, zorder=15)
    ax.text(14.97, 4.22, "Aggregate\nSimilarities\nto Class 1", ha='center', va='center',
            color='#991B1B', fontsize=7.8, fontweight='bold', zorder=25)
    ax.text(14.97, 3.75, "S_1 = mean(S^h_c1 / tau)", ha='center', va='center',
            color='#7F1D1D', fontsize=6.5, zorder=25)

    # Lines from Head pills and Prototype 1 to Aggregation 1
    for hy in head_ys:
        ax.plot([13.35, 13.95], [hy + 0.35, 4.12], color='#0F172A', lw=1.0, zorder=20)
    ax.annotate('', xy=(14.70, 4.70), xytext=(14.70, 2.23),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color='#0F172A', mutation_scale=12), zorder=22)

    # Scores S0 and S1 output arrows into Softmax (with clean text placement!)
    draw_arrow(ax, 16.00, 6.12, 16.25, 6.12, color='#0F172A', lw=1.4, zorder=20)
    ax.text(16.12, 6.42, "S_0", ha='center', va='center', color='#0F172A', fontsize=7.6, fontweight='bold', zorder=25)
    draw_arrow(ax, 16.00, 4.12, 16.25, 4.12, color='#0F172A', lw=1.4, zorder=20)
    ax.text(16.12, 4.42, "S_1", ha='center', va='center', color='#0F172A', fontsize=7.6, fontweight='bold', zorder=25)

    # Vertical Softmax Pill
    draw_3d_block(ax, 16.25, 3.45, 0.45, 3.40, depth=0.18, angle=35,
                  face_col='#6BA4D9', top_col='#91BFEF', side_col='#4682B4', edge_col='#1D4ED8', lw=1.1, zorder=15)
    ax.text(16.47, 5.15, "Softmax", ha='center', va='center',
            color='white', fontsize=8.8, fontweight='bold', rotation=90, zorder=25)

    # Mathematical Legend inside panel (Bottom Right)
    ax.text(15.45, 1.15, "H_q = Query Vector Heads\nP_c = Prototype Vectors\nS^h_c / tau = Head-wise Similarity / Temp",
            ha='center', va='center', color='#475569', fontsize=6.8, fontstyle='italic', zorder=25)

    # Arrow from Softmax out to Extracted Features
    draw_arrow(ax, 16.85, 5.15, 17.25, 5.15, color='#0F172A', lw=1.8, zorder=20)

    # -------------------------------------------------------------------------
    # COLUMN 6: EXTRACTED FEATURES (3D Silver Pillar)
    # -------------------------------------------------------------------------
    draw_3d_block(ax, 17.25, 2.40, 0.75, 5.20, depth=0.25, angle=35,
                  face_col='#E2E8F0', top_col='#F8FAFC', side_col='#CBD5E1', edge_col='#94A3B8', zorder=10)
    ax.text(17.62, 5.00, "Extracted Features (16-dim Embedding)", ha='center', va='center',
            color='#0F172A', fontsize=9.0, fontweight='bold', rotation=90, zorder=30)

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
