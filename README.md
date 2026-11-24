# deeplearningarchitecture

## Multimodal Machine Learning for Incident Adolescent Anxiety Prediction: An Exploratory Proof-of-Concept in an Isolated Twin Cohort

This repository contains the leak-free, reproducible machine learning codebase and evaluation suite for predicting incident adolescent anxiety in an isolated longitudinal twin cohort (Queensland Twin Adolescent Brain project, QTAB).

---

### Methodological Overview & Honest Framing

Predicting *incident* internalizing anxiety disorders in adolescents over a 2-year longitudinal follow-up (onset between Session 1 and Session 2) is a challenging task due to low base-rate prevalence ($12.5\%$, 38 cases across 304 participants) and subtle early prodromal manifestations.

Following a methodological audit, this codebase enforces strict zero-leakage protocols:
1. **Strict Family Isolation:** Twin pairs are assigned intact to partitions to prevent genetic and shared-environmental contamination (Train: 90 families, 180 subjects, 24 incident cases; Validation: 31 families, 62 subjects, 7 incident cases; Test: 31 families, 62 subjects, 7 incident cases).
2. **Train-Only Preprocessing:** All scalers, PCA projections ($k=5$ for sMRI), and classifiers are fit strictly on the training partition (`train.csv`). Zero validation or test data enters dimensionality reduction or model fitting.
3. **Pre-Specified Primary Model (Dawes' Rule of Equal Weighting):**
   - Fine-grained grid tuning on finite validation cohorts ($n_{\text{val}} = 62$, 7 events) overfits validation data (Validation AUC $0.720 \to$ Test AUC $0.665$, 95% CI $[0.461, 0.870]$ spanning chance).
   - The unweighted **Simple Average ($1/3, 1/3, 1/3$)** acts as a Bayesian uniform shrinkage prior (Dawes, 1979; Wainer, 1976), achieving held-out **Test ROC-AUC = 0.7377** (95% CI $[0.601, 0.881]$) and **Test F1 = 0.3226** (Sensitivity $71.43\%$, Specificity $65.45\%$, NPV $94.74\%$, Adjusted PPV $22.80\%$).
4. **Modality Performance Reality:**
   - **Module 1 (sMRI, Pretrained Med3D 3D-ResNet-18 + PCA-5):** Test ROC-AUC = **0.7221** (Val AUC = 0.5117). Features pre-extracted from structural T1w scans.
   - **Module 2 (Questionnaires, FT-ProtoNet):** Test ROC-AUC = **0.5351** (Val AUC = 0.7429).
   - **Module 3 (Phenotypic, FT-ProtoNet):** Test ROC-AUC = **0.5571** (Val AUC = 0.8091).
   - **Statistical Multiplicity:** Multimodal fusion is significantly better than chance ($p = 0.0020$, BH FDR $q = 0.0140$, Bonferroni $p = 0.0140$), but does *not* achieve statistically significant superiority over sMRI alone ($p = 0.911$).
5. **Feature Selection & Subpopulation Inversion:**
   - Automated train-only screening across total symptom severity scales (Pearson, ANOVA, Mutual Information, 5-Fold Nested CV) causes subpopulation inversion on held-out test data (Test AUC $\approx 0.32 - 0.40$). This occurs because chronic/persistent anxious controls exhibit high baseline symptoms while incident cases exhibit lower prodromal symptoms.
   - Transition-specific markers (parent-reported somatic complaints `pSDQ03`, muscle tension `SPHERE`, and parent-child symptom discrepancies `diff_26`, `diff_worry`) restore positive test orientation.
6. **Calibration Disclosure:**
   - Platt scaling achieves a Brier score of **0.0999** and near-zero ECE (**0.0003**), which matches the uninformative constant base-rate prediction ($7/62 = 11.3\%$, Brier = **0.1002**). This reflects probability shrinkage toward the cohort base rate rather than high-resolution individual risk discrimination.
7. **Resolution of Prototypical Network Training vs. Scoring Duality (Modules 2 & 3):**
   - **Training Episodes:** In each training epoch, class prototypes $\mathbf{c}_k$ are computed from a balanced $12$-shot support set ($|S_0|=12, |S_1|=12$). The episodic cross-entropy loss is evaluated on the leftover query set ($12$ incident vs $144$ control subjects).
   - **Unified Checkpointing & Inference:** Validation checkpoint selection evaluates validation subjects strictly using each episode's $12$-shot support prototypes $\mathbf{c}_k$. The optimal checkpoint saves the model weights along with the frozen support prototypes $\mathbf{c}_k^*$. Test-set risk scoring evaluates queries against these exact same support prototypes $\mathbf{c}_k^*$.
   - **Methodological Consistency:** The scoring duality is completely eliminated—test centers are built by the exact same $12$-shot support rule as training centers, and the episodic loss directly optimized the network representations against those exact prototype centers.

---

### Repository Structure

```
├── code/
│   ├── 01_mri_module.py             # Pretrained Med3D sMRI -> Train-Only PCA-5 -> Balanced Metric Head
│   ├── 02_questionnaire_module.py   # FT-ProtoNet on 9 transition vulnerability items
│   ├── 03_static_module.py          # FT-ProtoNet on 4 socioeconomic/pubertal indicators
│   ├── 04_multimodal_fusion.py      # Primary Simple Average, Grid Search, Latent, Calibration, Multiplicity
│   ├── 05_shap_analysis.py          # Model-agnostic SHAP interpretability on deep ProtoNet prototypes
│   ├── 06_feature_selection_audit.py# Empirical audit: Leaky vs Train-Only vs Nested CV vs Clinical
│   └── 07_repeated_family_cv.py     # 5x5 Repeated Family-Stratified CV (25 folds) with frozen train threshold
├── data/
│   ├── splits/                      # Family-isolated train.csv, val.csv, test.csv (zero overlap)
│   ├── raw/                         # Questionnaire, phenotypic, and outcome tables
│   ├── mri/                         # Pre-extracted 768-D Med3D embeddings (standalone__pretrained.npz)
│   └── processed/                   # Intermediate representations and out-of-fold probabilities
├── models/                          # Saved model weights (.pt, .pkl) and metadata
├── results/                         # Summary CSV tables, metrics JSONs, and 300 DPI publication figures
└── requirements.txt                 # Pinned dependencies
```

---

### Quick Start

```bash
# Clone the repository
git clone https://github.com/taosiff/deeplearningarchitecture.git
cd deeplearningarchitecture

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run Module 1 (sMRI)
python code/01_mri_module.py

# Run Module 2 (Questionnaires) and Module 3 (Phenotypic)
python code/02_questionnaire_module.py
python code/03_static_module.py

# Run Multimodal Fusion Suite & Statistical Analysis
python code/04_multimodal_fusion.py

# Run Feature Selection Audit & Repeated Cross-Validation
python code/06_feature_selection_audit.py
python code/07_repeated_family_cv.py
```
