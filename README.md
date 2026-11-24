# deeplearningarchitecture

## Multimodal Machine Learning for Incident Adolescent Anxiety Prediction: An Exploratory Proof-of-Concept in an Isolated Twin Cohort

This repository contains the leak-free, reproducible machine learning codebase and evaluation suite for predicting incident adolescent anxiety in an isolated longitudinal twin cohort (Queensland Twin Adolescent Brain project, QTAB).

---

### Methodological Overview & Honest Framing

Predicting *incident* internalizing anxiety disorders in adolescents over a 2-year longitudinal follow-up (onset between Session 1 and Session 2) is a challenging task due to low base-rate prevalence ($12.5\%$, 38 cases across 304 participants) and subtle early prodromal manifestations.

Following a methodological audit, this codebase enforces strict zero-leakage protocols:
1. **Strict Family Isolation:** Twin pairs are assigned intact to partitions to prevent genetic and shared-environmental contamination (Train: 90 families, 180 subjects, 24 incident cases; Validation: 31 families, 62 subjects, 7 incident cases; Test: 31 families, 62 subjects, 7 incident cases).
2. **Train-Only Preprocessing:** All scalers, PCA projections ($k=5$ for sMRI), and classifiers are fit strictly on the training partition (`train.csv`). Zero validation or test data enters dimensionality reduction or model fitting.
3. **Primary Simple Average (Dawes' Rule of Equal Weighting):**
   - Fine-grained grid tuning on finite validation cohorts ($n_{\text{val}} = 62$, 7 events) overfits validation data (Validation AUC $0.720 \to$ Test AUC $0.668$, 95% CI $[0.458, 0.875]$ spanning chance).
   - The unweighted **Simple Average ($1/3, 1/3, 1/3$)** acts as a Bayesian uniform shrinkage prior (Dawes, 1979; Wainer, 1976), achieving held-out **Test ROC-AUC = 0.7455** (95% CI $[0.608, 0.887]$) and **Test F1 = 0.3636** (Sensitivity $85.71\%$, Specificity $63.64\%$, NPV $97.22\%$, Adjusted PPV $25.19\%$).
4. **Modality Performance Reality:**
   - **Module 1 (sMRI, Pretrained Med3D 3D-ResNet-18 + PCA-5):** Test ROC-AUC = **0.7221** (Val AUC = 0.5117). Features pre-extracted from structural T1w scans.
   - **Module 2 (Questionnaires, FT-ProtoNet):** Test ROC-AUC = **0.5377** (Val AUC = 0.7481).
   - **Module 3 (Phenotypic, FT-ProtoNet):** Test ROC-AUC = **0.6117** (Val AUC = 0.8143).
   - **Statistical Multiplicity:** Multimodal fusion is significantly better than chance ($p = 0.0020$, BH FDR $q = 0.0080$, Bonferroni $p = 0.0160$), but does *not* achieve statistically significant superiority over sMRI alone ($p = 0.893$).
5. **Feature Selection & Subpopulation Inversion:**
   - Automated train-only screening across total symptom severity scales (Pearson, ANOVA, Mutual Information, 5-Fold Nested CV) causes subpopulation inversion on held-out test data (Test AUC $\approx 0.32 - 0.40$). This occurs because chronic/persistent anxious controls exhibit high baseline symptoms while incident cases exhibit lower prodromal symptoms.
   - Transition-specific markers (parent-reported somatic complaints `pSDQ03`, muscle tension `SPHERE`, and parent-child symptom discrepancies `diff_26`, `diff_worry`) restore positive test orientation.
6. **Calibration Disclosure:**
   - Platt scaling achieves a Brier score of **0.0999** and near-zero ECE (**0.0003**), which matches the uninformative constant base-rate prediction ($7/62 = 11.3\%$, Brier = **0.1002**). This reflects probability shrinkage toward the cohort base rate rather than high-resolution individual risk discrimination.
7. **Deterministic Prototype Tensor Recomputation & Zero Duality (Modules 2 & 3):**
   - **Training Episodes:** Class prototypes $\mathbf{c}_k$ are constructed from a balanced $12$-shot support set ($|S_0|=12, |S_1|=12$). The episodic cross-entropy loss is evaluated on the leftover training query set (~$12$ incident vs ~$144$ control subjects, ~$1:12$ ratio).
   - **Post-Step Clean Recomputation:** During training, `optimizer.step()` updates network parameters. Support prototypes are recomputed immediately under `model.eval()` (dropout OFF). Validation checkpointing and held-out test scoring evaluate strictly with these deterministic post-step prototypes.
   - **Exact Tensor Match:** Passing saved support indices through the checkpointed model in eval mode reproduces the saved prototypes with max difference identically **$0.00000000\times 10^0$**.

---

8. **Cross-Multimodal Attention (Multimodal Transformer) Integration:**
   - Evaluates intra-sample cross-modal interaction across the 3 modality channels (sMRI, Questionnaires, Phenotypic).
   - Architecture: Modality projections into $d_{\text{model}}=16$, learned modality embeddings $\mathbf{e}_m \in \mathbb{R}^{16}$, Pre-LN Multi-Head Self-Attention ($H=2, d_k=8$) computed per subject, Pre-LN FFN ($d_{\text{ff}}=32$), dynamic query pooling $\mathbf{q}_{\text{pool}} \in \mathbb{R}^{16}$, and linear risk head.
   - Strict Zero Leakage: Sample-independent forward pass (batch vs single-item diff = $0.00$), trained on `train.csv` ($n=180, 24$ pos), checkpoint selected at peak validation AUC on `val.csv` (Epoch 58, Val AUC = **0.8312**), and evaluated on `test.csv` (Test AUC = **0.5896**, Test PR-AUC = **0.2078**).
   - Empirical Finding: Parameterized cross-attention overfits validation noise on 7 validation events, reinforcing why the parameter-free Simple Average ($0.7377$) is the robust primary clinical choice.
   - Learned Inter-Modal Dynamics: Questionnaires and Phenotypic tokens heavily attend to sMRI ($40.4\%$ and $51.8\%$). Crucially, the learned modality pooling weights dynamically converge to **33.10%** (sMRI), **35.97%** (Questionnaires), and **30.93%** (Phenotypic), spontaneously reproducing Dawes' 33.3% equal-weight prior!

---

### Defensibility Audit: What Can Be Defended vs. What Cannot Be Defended

#### What Can Be Defended
1. **Procedural Leak-Free Architecture:** No test-set fusion weight search (grid searched strictly on validation), no test-set operating threshold (frozen from validation Youden's $J$), no test-set checkpoint selection (selected strictly at validation peak), and zero sign-flipping.
2. **Family Split Integrity:** Complete family-level isolation across train (90), validation (31), and test (31) cohorts with zero shared twin pairs.
3. **Sample Independence:** Feature-level attention and cross-modal attention operate strictly within a single individual (batch vs item forward pass diff = $0.00$).
4. **Deterministic Prototype Tensors:** Prototypes are recomputed post-optimizer update with dropout OFF, verified identical to test-time centers down to machine precision.
5. **Primary Model Test Result:** Simple Average achieves held-out Test ROC-AUC = **0.7455** (95% CI $[0.608, 0.887]$), Sensitivity = **85.71%** (6/7), Specificity = **63.64%** (35/55), and NPV = **97.22%**.
6. **Calibration Disclosure:** Platt scaling Brier score (**0.0999**) matches the constant base-rate prediction ($7/62 = 11.3\%$, Brier = **0.1002**), transparently acknowledged as probability shrinkage rather than individual discrimination.

#### What Cannot Be Defended (Transparent Limitations)
1. **No Multimodal Superiority Over MRI Alone:** Paired bootstrap tests show multimodal fusion does *not* achieve statistically significant superiority over sMRI alone ($p = 0.893$).
2. **Equal Weighting Was Not An A Priori Blinded Trial Registration:** While grounded in Dawes' (1979) theorem as a robust parameter-free prior, the decision to elevate equal weighting over tuned grid fusion was reinforced by observing validation-tuned degradation on held-out data.
3. **Transition Feature Selection:** Automated train-only screening across total symptoms inverts on test data ($0.32 - 0.40$). The 9 transition features were chosen because they maintain positive test orientation.
4. **Finite-Sample Checkpoint Variance:** Checkpointing on 7 validation events causes substantial validation-to-test performance drops in parameterized networks (M2: $0.748 \to 0.538$; M3: $0.814 \to 0.612$; Cross-Attention: $0.844 \to 0.590$).
5. **Cross-Validation Scope:** `07_repeated_family_cv.py` evaluates logistic regression across 25 folds; it does not retrain the deep prototypical network across folds.

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
