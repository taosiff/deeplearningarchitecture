# deeplearningarchitecture

## Multimodal Deep Few-Shot Metric Learning for Incident Adolescent Anxiety Prediction

This repository contains the complete, leak-free, reproducible deep learning codebase and evaluation suite for predicting incident adolescent anxiety in an isolated longitudinal twin cohort study.

---

### Pipeline Architecture & Modules

The pipeline integrates three core modalities with zero data leakage (strict family-isolated train, validation, and test splits):

1. **Module 1: Structural MRI (`code/01_mri_module.py`)**
   - Pretrained Med3D 3D-ResNet-18 backbone (768-D representation).
   - Unsupervised PCA dimensionality reduction ($k=5$) preserving structural variance.
   - Class-balanced linear metric head with L2 regularization.

2. **Module 2: Anxiety Symptomatology (`code/02_questionnaire_module.py`)**
   - **FT-ProtoNet**: Feature-Tokenizer Multi-Head Self-Attention Prototypical Network.
   - Sample-independent episodic few-shot metric learning tailored for severe class imbalance (~1:6.5).
   - 9 clinically guided transition vulnerability items (somatic complaints, muscle tension, parent-child discrepancy).

3. **Module 3: Static Phenotypic (`code/03_static_module.py`)**
   - **FT-ProtoNet** architecture applied to socioeconomic and pubertal transition features.
   - Metric learning on Euclidean prototype distances.

4. **Multimodal Fusion & Multiplicity Control (`code/04_multimodal_fusion.py`)**
   - Multiple fusion paradigms: Simple Average ($33/33/33$), Balanced Late ($25/45/30$), Intermediate Latent Concatenation ($37\text{-D}$), Grid Search ($186$ candidates), and Stacking Meta-Learner.
   - Full calibration suite: Platt Scaling vs Isotonic Regression with 10-bin Expected Calibration Error (ECE) and Brier scores.
   - Multiplicity control: Benjamini-Hochberg False Discovery Rate ($q$-values) and Bonferroni corrections across all hypotheses.
   - 2,000 cluster-bootstrap iterations at family level.

5. **Model-Agnostic SHAP Interpretability (`code/05_shap_analysis.py`)**
   - High-resolution summary and feature importance plots directly explaining deep ProtoNet prototypes.

6. **Feature Selection Leakage Audit (`code/06_feature_selection_audit.py`)**
   - Empirical comparison between Train+Val leaky selection, strict Train-only selection ($k \in [5, 40]$), 5-fold Nested-CV, and clinically guided transition selection.

7. **5x5 Repeated Family-Stratified CV (`code/07_repeated_family_cv.py`)**
   - 25 independent family-isolated folds across the 304-subject cohort using `StratifiedGroupKFold`.

---

### Quick Start

```bash
# Clone the repository
git clone https://github.com/taosiff/deeplearningarchitecture.git
cd deeplearningarchitecture

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run multimodal evaluation suite
python code/04_multimodal_fusion.py

# Run repeated family-stratified cross-validation
python code/07_repeated_family_cv.py
```
