#!/usr/bin/env python3
"""
Scientific 3D Architecture Figures Generator (Logical & Mathematical Alignment)
================================================================================
Completely aligned with the QTAB Prospective Anxiety Manuscript methods:
  - Fig 1: Overall Multimodal Framework — Parameter-Free Equal-Weight Decision Fusion
  - Fig 2: Module 1 — Structural Neuroimaging Pipeline (Med3D 3D-ResNet-18 + PCA-5 + LR)
  - Fig 3: Module 2 — Behavioral Questionnaires Pipeline (2-D Psychometric Factor Analysis + XGBoost)
  - Fig 4: Module 3 — Phenotypic & Autonomic Pipeline (ANOVA-10 + BorderlineSMOTE + ExtraTrees)

Design Principles:
  - Pure white background (#FFFFFF), NO dark outer boxes, NO muddy drop-shadows
  - Clean isometric 3D beveled blocks (top highlight, right shading)
  - Short, punchy, modular technical labels (never essay text)
  - Logical accuracy: NO "gradient flow" labels between independent modules
  - Real scientific creativity:
      * Embedded high-res axial brain MRI slices
      * 3D isometric CNN tensor feature slabs with channel tags
      * TRUE 2D Psychometric Latent Factor Biplot (F1 vs F2 scatter & centroids)
      * BorderlineSMOTE decision boundary scatter with synthetic star points
      * Depth-regularized decision tree branching diagrams
      * Vectorized clinical icons (brain, clipboard, vitals avatar, funnel, wave)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

OUT_DIR = Path(r"G:\RESEARCH\QTAB_ANXIETY_PROJECT\temp_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Typography settings
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#1E293B'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'

# -------------------------------------------------------------------------
# CORE HELPER: 3D ISOMETRIC BEVELED BLOCK (CLEAN, SHARP, NO MUDDY SHADOWS)
# -------------------------------------------------------------------------
def draw_3d_block(ax, x, y, w, h, depth=0.35, angle=35,
                  face_col='#1F618D', top_col='#5DADE2', side_col='#154360',
                  edge_col='#0E2F44', lw=1.1, zorder=10):
    """Draws a clean isometric 3D block with top highlight and right shading."""
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
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.14",
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
                fontsize=7.5, fontweight='bold', bbox=bbox_p, zorder=zorder+5)

# -------------------------------------------------------------------------
# VECTOR ICONS
# -------------------------------------------------------------------------
def draw_brain_vector(ax, cx, cy, s=0.30, col='white', zorder=35):
    """Draws a stylized white anatomical brain vector."""
    for dx, dy in [(-s*0.35, 0), (s*0.35, 0), (-s*0.25, s*0.3), (s*0.25, s*0.3)]:
        ax.add_patch(patches.Circle((cx + dx, cy + dy), s*0.38, facecolor='none', edgecolor=col, lw=1.4, zorder=zorder))
    ax.plot([cx, cx], [cy - s*0.4, cy + s*0.5], color=col, lw=1.5, zorder=zorder)
    ax.plot([cx - s*0.3, cx - s*0.1], [cy + s*0.1, cy + s*0.2], color=col, lw=1.2, zorder=zorder)
    ax.plot([cx + s*0.1, cx + s*0.3], [cy + s*0.1, cy + s*0.2], color=col, lw=1.2, zorder=zorder)

def draw_clipboard_vector(ax, cx, cy, s=0.30, col='white', zorder=35):
    """Draws a stylized clipboard vector with checklist."""
    w, h = s * 0.9, s * 1.2
    ax.add_patch(patches.Rectangle((cx - w/2, cy - h/2), w, h, facecolor='none', edgecolor=col, lw=1.4, zorder=zorder))
    ax.add_patch(patches.Rectangle((cx - w*0.3, cy + h*0.38), w*0.6, h*0.18, facecolor=col, edgecolor=col, lw=0.8, zorder=zorder))
    for dy in [-0.25, 0.05, 0.32]:
        ax.plot([cx - w*0.15, cx + w*0.3], [cy + dy*h*0.65, cy + dy*h*0.65], color=col, lw=1.2, zorder=zorder)
        ax.plot([cx - w*0.35, cx - w*0.25, cx - w*0.18], [cy + dy*h*0.65, cy + dy*h*0.65 - 0.04, cy + dy*h*0.65 + 0.05],
                color=col, lw=1.2, zorder=zorder)

def draw_user_vector(ax, cx, cy, s=0.30, col='white', zorder=35):
    """Draws a stylized user/avatar biomarker vector."""
    ax.add_patch(patches.Circle((cx, cy + s*0.38), s*0.32, facecolor='none', edgecolor=col, lw=1.4, zorder=zorder))
    ax.add_patch(patches.Arc((cx, cy - s*0.48), s*1.15, s*1.15, angle=0, theta1=35, theta2=145, edgecolor=col, lw=1.4, zorder=zorder))
    ax.plot([cx - s*0.35, cx - s*0.15, cx, cx + s*0.15, cx + s*0.35],
            [cy - s*0.2, cy - s*0.2, cy - s*0.05, cy - s*0.35, cy - s*0.2], color=col, lw=1.1, zorder=zorder)


# =========================================================================
# FIGURE 1: OVERALL MULTIMODAL PREDICTION FRAMEWORK
# =========================================================================
def build_fig1():
    print("Generating Fig 1: Overall Multimodal Framework...")
    fig, ax = plt.subplots(figsize=(15.2, 8.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 9.4)
    ax.axis('off')

    # Top Modules (3D Beveled Blocks) - Representing the 3 Independently Trained Modules
    # Module 1 (Teal/Blue - Structural Neuroimaging)
    draw_3d_block(ax, 0.7, 6.8, 4.0, 1.40, depth=0.35, angle=35,
                  face_col='#1F618D', top_col='#5DADE2', side_col='#154360', edge_col='#0B2535', zorder=10)
    draw_brain_vector(ax, 1.15, 7.50, s=0.30, col='#EBF5FB', zorder=30)
    ax.text(2.90, 7.65, "Module 1: MRI-BASED", ha='center', va='center',
            color='white', fontsize=11, fontweight='bold', zorder=30)
    ax.text(2.90, 7.30, "3D-ResNet-18 (Med3D) + PCA-5", ha='center', va='center',
            color='#D4E6F1', fontsize=8.5, zorder=30)

    # Module 2 (Emerald Green - Behavioral Questionnaires)
    draw_3d_block(ax, 5.6, 6.8, 4.0, 1.40, depth=0.35, angle=35,
                  face_col='#239B56', top_col='#58D68D', side_col='#196F3D', edge_col='#0E4B24', zorder=10)
    draw_clipboard_vector(ax, 6.05, 7.50, s=0.28, col='#E8F8F5', zorder=30)
    ax.text(7.85, 7.65, "Module 2: QUESTIONNAIRES", ha='center', va='center',
            color='white', fontsize=11, fontweight='bold', zorder=30)
    ax.text(7.85, 7.30, "2-D Factor Analysis + XGBoost", ha='center', va='center',
            color='#D1F2EB', fontsize=8.5, zorder=30)

    # Module 3 (Warm Rust/Orange - Phenotypic & Autonomic)
    draw_3d_block(ax, 10.5, 6.8, 4.0, 1.40, depth=0.35, angle=35,
                  face_col='#D35400', top_col='#F39C12', side_col='#935116', edge_col='#592804', zorder=10)
    draw_user_vector(ax, 10.95, 7.50, s=0.28, col='#FEF5E7', zorder=30)
    ax.text(12.70, 7.65, "Module 3: PHENOTYPIC", ha='center', va='center',
            color='white', fontsize=11, fontweight='bold', zorder=30)
    ax.text(12.70, 7.30, "ANOVA-10 + SMOTE + ExtraTrees", ha='center', va='center',
            color='#FDEBD0', fontsize=8.5, zorder=30)

    # Downward inference flow arrows (LOGICALLY: Unimodal Inference, NOT "Gradient Flow"!)
    draw_arrow(ax, 2.70, 6.8, 2.70, 5.85, color='#94A3B8', lw=1.6, zorder=12)
    draw_arrow(ax, 7.60, 6.8, 7.60, 5.85, color='#94A3B8', lw=1.6, zorder=12)
    draw_arrow(ax, 12.50, 6.8, 12.50, 5.85, color='#94A3B8', lw=1.6, zorder=12)
    ax.text(2.70, 6.32, "Unimodal Scoring", ha='center', va='center', color='#64748B', fontsize=7.8, fontstyle='italic', zorder=25)
    ax.text(7.60, 6.32, "Unimodal Scoring", ha='center', va='center', color='#64748B', fontsize=7.8, fontstyle='italic', zorder=25)
    ax.text(12.50, 6.32, "Unimodal Scoring", ha='center', va='center', color='#64748B', fontsize=7.8, fontstyle='italic', zorder=25)

    # Middle Row: Calibrated Probability Slabs (Silver/Grey 3D Beveled Bars)
    draw_3d_block(ax, 1.00, 5.1, 3.4, 0.65, depth=0.25, angle=35,
                  face_col='#EAEDED', top_col='#F8F9F9', side_col='#D5D8DC', edge_col='#BDC3C7', zorder=10)
    ax.text(2.70, 5.42, "P_MRI", ha='center', va='center', color='#1A252C', fontsize=11, fontweight='bold', zorder=30)

    draw_3d_block(ax, 5.90, 5.1, 3.4, 0.65, depth=0.25, angle=35,
                  face_col='#EAEDED', top_col='#F8F9F9', side_col='#D5D8DC', edge_col='#BDC3C7', zorder=10)
    ax.text(7.60, 5.42, "P_Quest", ha='center', va='center', color='#1A252C', fontsize=11, fontweight='bold', zorder=30)

    draw_3d_block(ax, 10.80, 5.1, 3.4, 0.65, depth=0.25, angle=35,
                  face_col='#EAEDED', top_col='#F8F9F9', side_col='#D5D8DC', edge_col='#BDC3C7', zorder=10)
    ax.text(12.50, 5.42, "P_Pheno", ha='center', va='center', color='#1A252C', fontsize=11, fontweight='bold', zorder=30)

    # Convergence arrows to Decision Fusion with fixed equal-weight badges (w = 1/3)
    draw_arrow(ax, 2.70, 5.1, 5.6, 3.85, color='#94A3B8', lw=1.8, badge='w = 1/3', badge_col='#1F618D', zorder=12)
    draw_arrow(ax, 7.60, 5.1, 7.60, 3.85, color='#94A3B8', lw=1.8, badge='w = 1/3', badge_col='#239B56', zorder=12)
    draw_arrow(ax, 12.50, 5.1, 9.6, 3.85, color='#94A3B8', lw=1.8, badge='w = 1/3', badge_col='#D35400', zorder=12)

    ax.text(3.70, 4.65, "Probability Flow", ha='center', va='center', color='#64748B', fontsize=7.8,
            fontstyle='italic', rotation=23, zorder=25)
    ax.text(11.50, 4.65, "Probability Flow", ha='center', va='center', color='#64748B', fontsize=7.8,
            fontstyle='italic', rotation=-23, zorder=25)

    # Multimodal Decision-Level Fusion 3D Bar (Royal Purple - LOGICALLY NAMED AS IN PAPER!)
    draw_3d_block(ax, 3.2, 2.85, 8.8, 1.00, depth=0.35, angle=35,
                  face_col='#6C3483', top_col='#A569BD', side_col='#4A235A', edge_col='#2C1337', zorder=10)
    ax.text(7.60, 3.52, "Equal-Weight Decision-Level Fusion", ha='center', va='center',
            color='white', fontsize=11.5, fontweight='bold', zorder=30)
    ax.text(7.60, 3.15, "P_multimodal = (1/3) * P_MRI + (1/3) * P_Quest + (1/3) * P_Pheno",
            ha='center', va='center', color='#F5EEF8', fontsize=9.2, zorder=30)

    # Downward arrow to Threshold Calibration
    draw_arrow(ax, 7.60, 2.85, 7.60, 2.20, color='#94A3B8', lw=1.6, zorder=12)
    ax.text(7.60, 2.52, "Calibrated Cut-Point", ha='center', va='center', color='#64748B', fontsize=7.8, fontstyle='italic', zorder=25)

    # Threshold Optimization 3D Bar (Rich Gold / Amber)
    draw_3d_block(ax, 4.6, 1.50, 6.0, 0.62, depth=0.25, angle=35,
                  face_col='#D4AC0D', top_col='#F4D03F', side_col='#9A7D0A', edge_col='#4D3E05', zorder=10)
    ax.text(7.60, 1.81, "Youden's J Threshold Calibration (tau = 0.3439)",
            ha='center', va='center', color='white', fontsize=9.5, fontweight='bold', zorder=30)

    # Downward arrow to Final Prediction
    draw_arrow(ax, 7.60, 1.50, 7.60, 0.95, color='#94A3B8', lw=1.6, zorder=12)

    # Final Clinical Decision 3D Bar (Teal / Deep Cyan)
    draw_3d_block(ax, 4.1, 0.15, 7.0, 0.78, depth=0.30, angle=35,
                  face_col='#148F77', top_col='#48C9B0', side_col='#0E6252', edge_col='#083E34', zorder=10)
    ax.text(7.60, 0.64, "Prospective Incident Anxiety Classification", ha='center', va='center',
            color='white', fontsize=11, fontweight='bold', zorder=30)
    ax.text(7.60, 0.35, "ROC-AUC = 0.8052 | Sensitivity = 85.7% | Specificity = 56.4% | NPV = 96.9%",
            ha='center', va='center', color='#E8F8F5', fontsize=8.6, fontweight='bold', zorder=30)

    fig.savefig(OUT_DIR / "Fig1.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Fig1.png generated in temp_figures/")


# =========================================================================
# FIGURE 2: MODULE 1 — STRUCTURAL NEUROIMAGING PIPELINE
# =========================================================================
def build_fig2():
    print("Generating Fig 2: Module 1 MRI Pipeline...")
    fig, ax = plt.subplots(figsize=(15.8, 8.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 15.8)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # 1. INPUT RAW SCANS (Stacked dark cards + Real axial brain slice thumbnail)
    draw_3d_block(ax, 0.35, 4.4, 1.8, 2.5, depth=0.22, angle=35,
                  face_col='#1E293B', top_col='#334155', side_col='#0F172A', edge_col='#020617', zorder=10)
    ax.text(1.25, 6.35, "Input\n(Raw T1w MRI)\n91x109x91", ha='center', va='center',
            color='white', fontsize=8.5, fontweight='bold', zorder=30)
    
    brain_slice_path = OUT_DIR / "brain_slices_crop.png"
    if brain_slice_path.exists():
        im = Image.open(brain_slice_path)
        im_crop = im.crop((im.width*0.15, im.height*0.1, im.width*0.9, im.height*0.95))
        imagebox = OffsetImage(im_crop, zoom=0.38)
        ab = AnnotationBbox(imagebox, (1.25, 5.1), frameon=False, zorder=25)
        ax.add_artist(ab)

    draw_arrow(ax, 2.30, 5.5, 2.75, 5.5, color='#475569', lw=1.8, zorder=15)

    # 2. PREPROCESSING PIPELINE (Slate Blue Panel)
    panel_prep = patches.FancyBboxPatch((2.75, 1.1), 3.2, 6.7, boxstyle="round,pad=0.15,rounding_size=0.2",
                                        facecolor='#344955', edgecolor='#232F34', lw=1.2, zorder=5)
    ax.add_patch(panel_prep)
    ax.text(4.35, 7.42, "Standardized MRI Preprocessing\n(Reproducible Python Pipeline)", ha='center', va='center',
            color='white', fontsize=9.8, fontweight='bold', zorder=25)

    prep_steps = [
        ("Skull Stripping (Otsu)", 6.2, "skull"),
        ("Bias Correction (N4)", 4.9, "pulse"),
        ("Registration (MNI152)", 3.6, "brain"),
        ("Intensity Z-score", 2.3, "chart")
    ]
    for title, py, itype in prep_steps:
        pill = patches.FancyBboxPatch((3.05, py - 0.42), 2.6, 0.85, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor='#ECEFF1', edgecolor='#CFD8DC', lw=1.0, zorder=7)
        ax.add_patch(pill)
        if itype == 'skull':
            ax.add_patch(patches.Circle((3.40, py), 0.16, facecolor='#263238', edgecolor='none', zorder=15))
        elif itype == 'pulse':
            ax.plot([3.25, 3.35, 3.40, 3.45, 3.55], [py, py, py+0.18, py-0.18, py], color='#0284C7', lw=1.6, zorder=15)
        elif itype == 'brain':
            ax.add_patch(patches.Circle((3.40, py), 0.15, facecolor='none', edgecolor='#0D9488', lw=1.5, zorder=15))
        elif itype == 'chart':
            ax.add_patch(patches.Rectangle((3.28, py-0.15), 0.07, 0.22, facecolor='#64748B', zorder=15))
            ax.add_patch(patches.Rectangle((3.38, py-0.15), 0.07, 0.30, facecolor='#64748B', zorder=15))
            ax.add_patch(patches.Rectangle((3.48, py-0.15), 0.07, 0.16, facecolor='#64748B', zorder=15))

        ax.text(4.45, py, title, ha='center', va='center', color='#263238', fontsize=8.0, fontweight='bold', zorder=25)

    draw_arrow(ax, 6.05, 4.5, 6.55, 4.5, color='#475569', lw=1.8, zorder=15)

    # 3. MED3D RESNET-18 BACKBONE (Large Center Panel)
    panel_cnn = patches.FancyBboxPatch((6.55, 1.1), 4.9, 6.7, boxstyle="round,pad=0.15,rounding_size=0.2",
                                       facecolor='#344955', edgecolor='#232F34', lw=1.2, zorder=5)
    ax.add_patch(panel_cnn)
    ax.text(9.00, 7.42, "Med3D 3D-ResNet-18 Backbone\n(Pretrained SSL Checkpoint)", ha='center', va='center',
            color='white', fontsize=10.2, fontweight='bold', zorder=25)

    # Inner container for feature map tensor slabs
    inner_box = patches.FancyBboxPatch((6.8, 3.75), 4.4, 3.1, boxstyle="round,pad=0.1,rounding_size=0.15",
                                       facecolor='#ECEFF1', edgecolor='#CFD8DC', lw=1.0, zorder=6)
    ax.add_patch(inner_box)

    # 3D Feature Map Slabs with channel/resolution tags
    slabs = [
        (7.05, 4.30, 0.35, 1.9, '#3B82F6', '#93C5FD', '#1D4ED8', '#1E3A8A', 'Conv1\n64ch'),
        (7.70, 4.45, 0.40, 1.5, '#06B6D4', '#67E8F9', '#0E7490', '#155E75', 'Res2\n128ch'),
        (8.40, 4.60, 0.45, 1.2, '#6366F1', '#A5B4FC', '#4338CA', '#312E81', 'Res3\n256ch'),
        (9.15, 4.75, 0.50, 0.9, '#8B5CF6', '#C4B5FD', '#6D28D9', '#4C1D95', 'Res4\n512ch')
    ]
    for sx, sy, sw, sh, fc, tc, sc, ec, tag in slabs:
        draw_3d_block(ax, sx, sy, sw, sh, depth=0.3, angle=35,
                      face_col=fc, top_col=tc, side_col=sc, edge_col=ec, zorder=10)
        ax.text(sx + sw/2, sy - 0.28, tag, ha='center', va='top', color='#334155', fontsize=6.8, fontweight='bold', zorder=25)

    # Global AvgPool & 768-D segmented bar
    draw_3d_block(ax, 10.15, 4.25, 0.22, 1.9, depth=0.18, angle=35,
                  face_col='#334155', top_col='#64748B', side_col='#0F172A', edge_col='#020617', zorder=10)
    for cy in np.linspace(4.35, 5.95, 7):
        ax.plot([10.15, 10.37], [cy, cy], color='#94A3B8', lw=0.8, zorder=15)
    ax.text(10.65, 5.2, "768-dim\nvector", ha='left', va='center', color='#0F172A', fontsize=7.2, fontweight='bold', zorder=25)

    # Downward flow to PCA feature selection funnel
    draw_arrow(ax, 9.00, 3.75, 9.00, 3.05, color='#0284C7', lw=1.8, zorder=15)

    # Feature selection container
    fs_box = patches.FancyBboxPatch((6.8, 1.4), 4.4, 1.6, boxstyle="round,pad=0.1,rounding_size=0.15",
                                    facecolor='#ECEFF1', edgecolor='#CFD8DC', lw=1.0, zorder=6)
    ax.add_patch(fs_box)
    
    # Funnel graphic on left
    funnel_pts = np.array([[7.1, 2.7], [7.7, 2.7], [7.45, 2.0], [7.35, 2.0]])
    funnel = patches.Polygon(funnel_pts, closed=True, facecolor='none', edgecolor='#0284C7', lw=1.8, zorder=15)
    ax.add_patch(funnel)
    ax.plot([7.4, 7.4], [2.0, 1.65], color='#0284C7', lw=1.8, zorder=15)

    ax.text(8.8, 2.45, "Orthogonal PCA Reduction", ha='center', va='center',
            color='#0F172A', fontsize=8.8, fontweight='bold', zorder=25)
    ax.text(8.8, 2.0, "Pool StandardScaler -> 5 Components\nExplaining 62.8% Variance (Z_mri in R^5)",
            ha='center', va='center', color='#475569', fontsize=7.5, zorder=25)

    draw_arrow(ax, 11.55, 4.5, 12.05, 4.5, color='#475569', lw=1.8, zorder=15)

    # 4. SUPERVISED CLASSIFIER PANEL (Right)
    panel_clf = patches.FancyBboxPatch((12.05, 1.1), 3.3, 6.7, boxstyle="round,pad=0.15,rounding_size=0.2",
                                       facecolor='#344955', edgecolor='#232F34', lw=1.2, zorder=5)
    ax.add_patch(panel_clf)
    ax.text(13.7, 7.42, "Cost-Sensitive Probe\n& Evaluation", ha='center', va='center',
            color='white', fontsize=10.2, fontweight='bold', zorder=25)

    # Model card
    draw_pill(ax, 12.35, 5.2, 2.7, 1.3, text="L2-Regularized\nLogistic Regression\n(C = 0.1, balanced)",
              face_col='#ECEFF1', edge_col='#CFD8DC', text_col='#0F172A', font_size=8.2, font_weight='bold')
    
    draw_arrow(ax, 13.7, 5.2, 13.7, 4.35, color='#0284C7', lw=1.6, zorder=15)

    # Probability output card
    draw_pill(ax, 12.35, 3.25, 2.7, 1.0, text="Output Probability:\nP_MRI in [0, 1]",
              face_col='#E0F2FE', edge_col='#0284C7', text_col='#0369A1', font_size=8.8, font_weight='bold')

    # Metrics pill
    draw_pill(ax, 12.35, 1.55, 2.7, 1.3,
              text="Test ROC-AUC: 0.7091\nPR-AUC: 0.3186\nSensitivity: 71.4%\nSpecificity: 58.2%",
              face_col='#0F172A', edge_col='#334155', text_col='#38BDF8', font_size=7.8, font_weight='bold')

    fig.savefig(OUT_DIR / "Fig2.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Fig2.png generated in temp_figures/")


# =========================================================================
# FIGURE 3: MODULE 2 — PSYCHIATRIC QUESTIONNAIRES PIPELINE (LOGICALLY ALIGNED!)
# =========================================================================
def build_fig3():
    print("Generating Fig 3: Module 2 Questionnaire Pipeline (FA + XGBoost)...")
    fig, ax = plt.subplots(figsize=(15.8, 8.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 15.8)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # 1. 3D GLASS CLIPBOARD / INPUT (Left)
    draw_3d_block(ax, 0.4, 4.3, 1.9, 2.6, depth=0.35, angle=35,
                  face_col='#239B56', top_col='#58D68D', side_col='#196F3D', edge_col='#0E4B24', zorder=10)
    draw_clipboard_vector(ax, 1.35, 5.85, s=0.35, col='#E8F8F5', zorder=30)
    ax.text(1.35, 4.85, "109 Raw Items\nSCAS, SMFQ, SDQ", ha='center', va='center',
            color='white', fontsize=8.2, fontweight='bold', zorder=30)

    draw_arrow(ax, 2.45, 5.5, 3.0, 5.5, color='#239B56', lw=1.8, zorder=15)

    # 2. PREPROCESSING & PURGING
    draw_pill(ax, 3.0, 4.4, 2.3, 2.2,
              text="Leakage-Aware\nPurging\n\n- Purge sum totals\n- In-fold imputation\n- Pool StandardScaler\n(Zero proxy leakage)",
              face_col='#FFFFFF', edge_col='#239B56', text_col='#0F172A', font_size=7.8, font_weight='normal')

    draw_arrow(ax, 5.4, 5.5, 5.9, 5.5, color='#239B56', lw=1.8, zorder=15)

    # 3. FACTOR ANALYSIS ENGINE (Center Panel)
    panel_fa = patches.FancyBboxPatch((5.9, 1.1), 4.3, 6.7, boxstyle="round,pad=0.15,rounding_size=0.2",
                                      facecolor='#FFFFFF', edgecolor='#239B56', lw=1.5, zorder=5)
    ax.add_patch(panel_fa)
    ax.text(8.05, 7.42, "Psychometric Factor Analysis", ha='center', va='center',
            color='#239B56', fontsize=10.5, fontweight='bold', zorder=25)
    ax.text(8.05, 7.08, "Matrix Decomposition: X - mu = L*F + eps", ha='center', va='center',
            color='#64748B', fontsize=7.8, fontstyle='italic', zorder=25)

    # 3D Latent Factor Slabs (Factor 1 and Factor 2)
    draw_3d_block(ax, 6.2, 5.45, 3.7, 1.10, depth=0.25, angle=35,
                  face_col='#E0F2FE', top_col='#F0F9FF', side_col='#BAE6FD', edge_col='#0284C7', zorder=10)
    ax.text(8.05, 6.12, "Factor 1: General Distress", ha='center', va='center',
            color='#0369A1', fontsize=9.2, fontweight='bold', zorder=30)
    ax.text(8.05, 5.72, "Pervasive worry, dysphoria, affective tension", ha='center', va='center',
            color='#475569', fontsize=7.2, zorder=30)

    draw_3d_block(ax, 6.2, 3.95, 3.7, 1.10, depth=0.25, angle=35,
                  face_col='#F3E8FF', top_col='#FAF5FF', side_col='#E9D5FF', edge_col='#7C3AED', zorder=10)
    ax.text(8.05, 4.62, "Factor 2: Somatic Tension", ha='center', va='center',
            color='#6D28D9', fontsize=9.2, fontweight='bold', zorder=30)
    ax.text(8.05, 4.22, "Physical complaints, social-evaluative fear", ha='center', va='center',
            color='#475569', fontsize=7.2, zorder=30)

    # Clean projection arrow into continuous latent representation
    ax.annotate('', xy=(7.0, 3.82), xytext=(9.1, 3.82),
                arrowprops=dict(arrowstyle="->", lw=1.3, color='#7C3AED', connectionstyle="arc3,rad=-0.35"), zorder=20)
    
    ax.text(8.05, 3.10, "Continuous Latent Factors: Z_quest = [F1, F2] in R^2",
            ha='center', va='center', color='#239B56', fontsize=7.6, fontweight='bold', zorder=25)

    # Bottom Methodological Pillars (3D Beveled sub-plates)
    draw_3d_block(ax, 6.25, 1.55, 1.05, 0.75, depth=0.15, angle=35,
                  face_col='#FFFFFF', top_col='#F8FAFC', side_col='#E2E8F0', edge_col='#94A3B8', zorder=10)
    ax.text(6.77, 1.92, "Varimax\nRotation", ha='center', va='center', color='#0F172A', fontsize=7.5, fontweight='bold', zorder=30)

    draw_3d_block(ax, 7.52, 1.55, 1.05, 0.75, depth=0.15, angle=35,
                  face_col='#FFFFFF', top_col='#F8FAFC', side_col='#E2E8F0', edge_col='#94A3B8', zorder=10)
    ax.text(8.05, 1.92, "In-Fold\nImpute", ha='center', va='center', color='#0F172A', fontsize=7.5, fontweight='bold', zorder=30)

    draw_3d_block(ax, 8.80, 1.55, 1.05, 0.75, depth=0.15, angle=35,
                  face_col='#FFFFFF', top_col='#F8FAFC', side_col='#E2E8F0', edge_col='#94A3B8', zorder=10)
    ax.text(9.32, 1.92, "spw = 6.0\nScale", ha='center', va='center', color='#0F172A', fontsize=7.5, fontweight='bold', zorder=30)

    draw_arrow(ax, 10.3, 5.0, 10.8, 5.0, color='#239B56', lw=1.8, zorder=15)

    # 4. 2-D PSYCHOMETRIC LATENT FACTOR BIPLOT (LOGICALLY ACCURATE: F1 vs F2!)
    biplot_box = patches.FancyBboxPatch((10.8, 3.4), 2.3, 3.2, boxstyle="round,pad=0.1,rounding_size=0.15",
                                        facecolor='#FFFFFF', edgecolor='#239B56', lw=1.2, zorder=6)
    ax.add_patch(biplot_box)
    ax.text(11.95, 6.25, "Latent Biplot (F1 vs F2)", ha='center', va='center',
            color='#239B56', fontsize=8.2, fontweight='bold', zorder=25)

    # 2D Cartesian Coordinate Axes for F1 and F2
    bx, by = 11.95, 4.8
    ax.plot([bx-0.85, bx+0.85], [by-0.3, by-0.3], color='#94A3B8', lw=1.2, zorder=15) # F1 axis
    ax.plot([bx-0.5, bx-0.5], [by-0.7, by+0.8], color='#94A3B8', lw=1.2, zorder=15)   # F2 axis
    ax.text(bx+0.7, by-0.45, "F1", ha='center', va='center', color='#64748B', fontsize=6.8, fontweight='bold', zorder=25)
    ax.text(bx-0.65, by+0.7, "F2", ha='center', va='center', color='#64748B', fontsize=6.8, fontweight='bold', zorder=25)

    # Scatter points in (F1, F2) latent space
    np.random.seed(42)
    # Healthy Controls (low F1, low F2)
    pts_neg_x = np.random.normal(bx - 0.2, 0.18, 14)
    pts_neg_y = np.random.normal(by - 0.05, 0.16, 14)
    ax.scatter(pts_neg_x, pts_neg_y, s=24, color='#10B981', edgecolors='#047857', lw=0.8, zorder=25)

    # Incident Anxiety Cases (elevated F1, elevated F2)
    pts_pos_x = np.random.normal(bx + 0.45, 0.16, 7)
    pts_pos_y = np.random.normal(by + 0.38, 0.16, 7)
    ax.scatter(pts_pos_x, pts_pos_y, s=28, color='#EF4444', edgecolors='#991B1B', lw=0.8, zorder=25)

    # Diamond Centroids
    ax.scatter([bx - 0.2], [by - 0.05], marker='D', s=45, color='#047857', edgecolors='white', lw=1.0, zorder=30)
    ax.scatter([bx + 0.45], [by + 0.38], marker='D', s=45, color='#B91C1C', edgecolors='white', lw=1.0, zorder=30)

    ax.text(11.95, 3.65, "Green: Control | Red: Anxiety", ha='center', va='center',
            color='#475569', fontsize=6.8, zorder=25)

    draw_arrow(ax, 13.2, 5.0, 13.6, 5.0, color='#239B56', lw=1.8, zorder=15)

    # 5. COST-SENSITIVE XGBOOST CLASSIFIER (Right Panel)
    panel_xgb = patches.FancyBboxPatch((13.6, 1.8), 2.0, 5.0, boxstyle="round,pad=0.1,rounding_size=0.15",
                                       facecolor='#239B56', edgecolor='#196F3D', lw=1.2, zorder=6)
    ax.add_patch(panel_xgb)
    ax.text(14.6, 6.4, "Cost-Sensitive\nXGBoost", ha='center', va='center',
            color='white', fontsize=9.5, fontweight='bold', zorder=25)
    ax.text(14.6, 5.4, "- spw = 6.0\n- depth = 3\n- lr = 0.1", ha='center', va='center',
            color='#D1FAE5', fontsize=8.2, zorder=25)

    # Output pill
    draw_pill(ax, 13.75, 4.0, 1.7, 0.85, text="P_Quest in [0, 1]",
              face_col='#FFFFFF', edge_col='#196F3D', text_col='#239B56', font_size=8.0, font_weight='bold')

    # Metrics
    ax.text(14.6, 2.7, "Test AUC: 0.7442\nSpecificity: 89.1%\n(Clinical Shield)", ha='center', va='center',
            color='#FEF08A', fontsize=7.8, fontweight='bold', zorder=25)

    fig.savefig(OUT_DIR / "Fig3.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Fig3.png generated in temp_figures/")


# =========================================================================
# FIGURE 4: MODULE 3 — PHENOTYPIC & AUTONOMIC PIPELINE
# =========================================================================
def build_fig4():
    print("Generating Fig 4: Module 3 Phenotypic Pipeline...")
    fig, ax = plt.subplots(figsize=(15.8, 8.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 15.8)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # 1. CANDIDATE BASELINE CARD (Left)
    draw_3d_block(ax, 0.4, 4.3, 1.9, 2.6, depth=0.25, angle=35,
                  face_col='#D35400', top_col='#F39C12', side_col='#935116', edge_col='#592804', zorder=10)
    draw_user_vector(ax, 1.35, 5.85, s=0.35, col='#FEF5E7', zorder=30)
    ax.text(1.35, 4.85, "76 Measures\nVitals, SES,\nBullying, Cog.", ha='center', va='center',
            color='white', fontsize=8.2, fontweight='bold', zorder=30)

    draw_arrow(ax, 2.40, 5.5, 2.85, 5.5, color='#D35400', lw=1.8, zorder=15)

    # 2. ROBUST NORMALIZATION PILL
    draw_pill(ax, 2.85, 4.5, 2.3, 2.0,
              text="RobustScaler\nx_rob = (x-med)/IQR\n\n- In-fold median\n- Removes outlier\n  distortion in vitals",
              face_col='#FFFFFF', edge_col='#D35400', text_col='#0F172A', font_size=8.0, font_weight='normal')

    draw_arrow(ax, 5.25, 5.5, 5.75, 5.5, color='#D35400', lw=1.8, zorder=15)

    # 3. ANOVA-10 FEATURE SCREENING & BORDERLINESMOTE (Center Panel)
    panel_center = patches.FancyBboxPatch((5.75, 1.1), 4.6, 6.7, boxstyle="round,pad=0.15,rounding_size=0.2",
                                          facecolor='#FFFFFF', edgecolor='#D35400', lw=1.5, zorder=5)
    ax.add_patch(panel_center)
    ax.text(8.05, 7.42, "ANOVA-10 Screen & Boundary Synthesis", ha='center', va='center',
            color='#D35400', fontsize=10.5, fontweight='bold', zorder=25)

    # Ranked Features List (Horizontal mini-bars)
    bars_data = [
        ("Resting Pulse Rate", 0.95),
        ("Systolic Blood Pressure", 0.88),
        ("Birth Advantage (IER)", 0.80),
        ("Birth Disadvantage (IRSD)", 0.74),
        ("Peer Bullying Victimization", 0.68),
        ("Cognitive CardSort", 0.60)
    ]
    for idx, (name, val) in enumerate(bars_data):
        by = 6.65 - idx * 0.42
        ax.add_patch(patches.Rectangle((6.05, by - 0.12), 3.0, 0.24, facecolor='#F1F5F9', zorder=10))
        ax.add_patch(patches.Rectangle((6.05, by - 0.12), 3.0 * val, 0.24, facecolor='#EA580C', alpha=0.85, zorder=12))
        ax.text(6.15, by, name, ha='left', va='center',
                color='white' if val > 0.4 else '#0F172A', fontsize=7.2, fontweight='bold', zorder=25)

    ax.text(8.05, 4.05, "Top-10 Predictive Biomarkers (Z_pheno in R^10)", ha='center', va='center',
            color='#9A3412', fontsize=8.0, fontweight='bold', zorder=25)

    # 4. BORDERLINESMOTE BOUNDARY SCATTER GRAPHIC (Scientific visual element!)
    scatter_box = patches.FancyBboxPatch((6.05, 1.45), 4.0, 2.3, boxstyle="round,pad=0.1,rounding_size=0.15",
                                         facecolor='#FFF7ED', edgecolor='#FB923C', lw=1.0, zorder=6)
    ax.add_patch(scatter_box)
    ax.text(8.05, 3.45, "BorderlineSMOTE (k = 5 Boundary Synthesis)", ha='center', va='center',
            color='#D35400', fontsize=8.2, fontweight='bold', zorder=25)

    # Decision boundary line
    ax.plot([6.4, 9.7], [1.8, 3.1], color='#DC2626', lw=1.2, linestyle='--', zorder=15)

    # Scatter points
    np.random.seed(10)
    ax.scatter(np.random.uniform(6.3, 7.8, 10), np.random.uniform(2.2, 3.1, 10),
               s=25, color='#3B82F6', edgecolors='#1D4ED8', zorder=20)
    ax.scatter(np.random.uniform(8.3, 9.7, 6), np.random.uniform(1.7, 2.4, 6),
               s=28, color='#EF4444', edgecolors='#B91C1C', zorder=20)
    # Synthetic boundary points (magenta stars)
    ax.scatter([7.8, 8.2, 8.6], [2.35, 2.50, 2.65], marker='*', s=55, color='#EC4899', edgecolors='#BE185D', lw=0.8, zorder=25)

    ax.text(8.05, 1.68, "Blue: Control | Red: Case | Magenta: Synthetic Boundary Points",
            ha='center', va='center', color='#475569', fontsize=6.8, zorder=25)

    draw_arrow(ax, 10.45, 4.5, 10.95, 4.5, color='#D35400', lw=1.8, zorder=15)

    # 5. DEPTH-REGULARIZED EXTRATREES CLASSIFIER (Right Panel)
    panel_et = patches.FancyBboxPatch((10.95, 1.6), 4.4, 5.7, boxstyle="round,pad=0.15,rounding_size=0.2",
                                      facecolor='#1E1B4B', edgecolor='#0F172A', lw=1.2, zorder=6)
    ax.add_patch(panel_et)
    ax.text(13.15, 6.85, "Depth-Regularized ExtraTrees", ha='center', va='center',
            color='white', fontsize=10.2, fontweight='bold', zorder=25)
    ax.text(13.15, 6.45, "Ensemble of 300 Randomized Trees", ha='center', va='center',
            color='#C7D2FE', fontsize=8.0, zorder=25)

    # Draw stylized tree diagrams (3 mini trees)
    for tx in [11.8, 13.15, 14.5]:
        ax.add_patch(patches.Circle((tx, 5.6), 0.12, facecolor='#818CF8', zorder=20))
        ax.plot([tx, tx-0.35], [5.6, 5.0], color='#C7D2FE', lw=1.1, zorder=15)
        ax.plot([tx, tx+0.35], [5.6, 5.0], color='#C7D2FE', lw=1.1, zorder=15)
        ax.add_patch(patches.Circle((tx-0.35, 5.0), 0.09, facecolor='#A5B4FC', zorder=20))
        ax.add_patch(patches.Circle((tx+0.35, 5.0), 0.09, facecolor='#A5B4FC', zorder=20))

    ax.text(13.15, 4.45, "Randomized Cuts (max_depth = 3)", ha='center', va='center',
            color='#E0E7FF', fontsize=8.0, zorder=25)

    # Output pill
    draw_pill(ax, 11.9, 3.4, 2.5, 0.85, text="P_Pheno in [0, 1]",
              face_col='#FFFFFF', edge_col='#4338CA', text_col='#1E1B4B', font_size=8.5, font_weight='bold')

    # Metrics
    ax.text(13.15, 2.35, "Test ROC-AUC: 0.7455\nPR-AUC: 0.3572\nSensitivity: 42.9% | Specificity: 78.2%",
            ha='center', va='center', color='#FEF08A', fontsize=8.0, fontweight='bold', zorder=25)

    fig.savefig(OUT_DIR / "Fig4.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Fig4.png generated in temp_figures/")


if __name__ == "__main__":
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    print("ALL 4 FIGURES SUCCESSFULLY REGENERATED IN temp_figures/")
