# Enzyme Function Classification (EC 1–6) using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green.svg)](https://lightgbm.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/tests-13%2F13%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Tesfalegnp%2FEnzyme--Function--Classification-blue?logo=github)](https://github.com/Tesfalegnp/Enzyme-Function-Classification)

A production-grade, scientifically rigorous machine learning pipeline for classifying primary enzyme functions (**EC 1 through EC 6**) from amino-acid sequences using the SwissProt-EC benchmark dataset ($212,273$ curated sequences).

---

## 🔬 Project Overview

Enzymes catalyze vital biological biochemical processes and are classified hierarchically by the **Enzyme Commission (EC)** system into six primary classes:
1. **EC 1 — Oxidoreductases**: Oxidation-reduction reactions
2. **EC 2 — Transferases**: Transfer of functional groups
3. **EC 3 — Hydrolases**: Hydrolysis bond cleavage
4. **EC 4 — Lyases**: Non-hydrolytic, non-oxidative elimination reactions
5. **EC 5 — Isomerases**: Geometric or structural isomerizations
6. **EC 6 — Ligases**: Joining two molecules coupled with ATP hydrolysis

This repository provides an end-to-end framework featuring:
- **Leakage-Free Multi-Stage Engineering**: Strict deduplication, conflict pruning, and stratified 70/15/15 train-validation-test partitioning.
- **Multi-Scale Biological Representations**: Amino Acid Composition (AAC 20D), Dipeptide Composition (DPC 400D), and 128D Truncated SVD embeddings of sparse 8,000D Tripeptide Composition (TPC), filtered to 500 optimal features via ANOVA F-value selection (`SelectKBest`) fitted strictly inside training folds.
- **Scalable Algorithmic Modeling**: Random Forest, LightGBM, and an $\mathcal{O}(N \cdot D)$ scalable `LinearSVC` with Platt scaling calibration (`CalibratedClassifierCV`) replacing intractable RBF SVM kernels.
- **Advanced Ensemble Architectures**: Majority Hard Voting, Calibrated Soft Voting, and Stacking with out-of-fold (OOF) cross-validated meta-features and a `LogisticRegression` meta-learner.
- **Comprehensive Error Analysis & Visualization**: Normalized confusion matrices, multi-class OvR ROC curves, Precision-Recall curves, sequence-length error stratification, and model disagreement mapping.

---

## 📊 Benchmark Results

All evaluations were benchmarked using **Macro F1** as the primary decision metric alongside **Matthews Correlation Coefficient (MCC)**, **Accuracy**, and **Macro AUPRC**.

### Executive Performance: Validation vs. Held-Out Test Set

| Model / Ensemble Architecture | Accuracy (Val) | Macro F1 (Val) | MCC (Val) | Accuracy (Test) | Macro F1 (Test)* | MCC (Test) | Macro AUPRC (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (100 trees, depth 20) | 0.7858 | 0.7939 | 0.7189 | 0.7884 | **0.7945** | 0.7222 | 0.8713 |
| **LightGBM** (100 trees, 31 leaves) | 0.7378 | 0.7420 | 0.6666 | 0.7395 | 0.7428 | 0.6688 | 0.8442 |
| **LinearSVC** ($C=0.05$, Platt-calibrated) | 0.5575 | 0.5351 | 0.4399 | 0.5565 | 0.5359 | 0.4402 | 0.5916 |
| **Hard Voting Ensemble** (Majority rule) | 0.7729 | 0.7766 | 0.7044 | 0.7744 | 0.7776 | 0.7065 | N/A** |
| **Soft Voting Ensemble** (Equal weights) | 0.7543 | 0.7599 | 0.6747 | 0.7587 | 0.7645 | 0.6807 | 0.8448 |
| **Stacking Classifier** (LR Meta-learner) | **0.7926** | **0.7905** | **0.7362** | **0.7977** | **0.7942** | **0.7426** | **0.9016** |

*\*Primary evaluation metric.*  
*\*\*Hard voting outputs discrete class votes; continuous probability rankings are not defined.*

### Per-Class Test Performance (Stacking Ensemble)

| Class | Enzyme Category | Precision | Recall | F1-Score | Support | ROC-AUC | AUPRC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EC 1** | Oxidoreductases | 0.8037 | 0.7711 | 0.7871 | 4,168 | 0.9687 | 0.8994 |
| **EC 2** | Transferases | 0.7601 | 0.8540 | 0.8043 | 11,790 | 0.9381 | 0.9023 |
| **EC 3** | Hydrolases | 0.8354 | 0.8491 | 0.8422 | 7,559 | 0.9427 | 0.8931 |
| **EC 4** | Lyases | 0.7699 | 0.6337 | 0.6952 | 2,943 | 0.9458 | 0.7423 |
| **EC 5** | Isomerases | 0.8148 | 0.7718 | 0.7927 | 1,709 | 0.9879 | 0.9575 |
| **EC 6** | Ligases | 0.8876 | 0.8848 | 0.8862 | 3,672 | 0.9897 | 0.9576 |
| **Macro Avg** | — | **0.8119** | **0.7941** | **0.7942** | **31,841** | **0.9622** | **0.9016** |

---

## 🏗️ Repository Architecture

```text
Enzyme-Function-Classification/
│
├── notebooks/
│   └── enzyme_function_classification.ipynb  # Primary presentation artifact (40 cells, all outputs preserved)
│
├── src/
│   ├── config.py           # Paths, seed (42), dimensions, and biological constants
│   ├── data_loader.py      # Robust parquet loader and schema validation
│   ├── data_validation.py  # Phase 1 diagnostic and cross-split leakage checks
│   ├── preprocessing.py    # Sequence cleaning, EC extraction, and deduplication
│   ├── splitting.py        # Stratified 70/15/15 partitioning (train/val/test)
│   ├── features.py         # AAC (20D), DPC (400D), TPC (8000D), SVD (128D), SelectKBest
│   ├── models.py           # RF, LightGBM, Scalable LinearSVC, 5-fold CV, tuning
│   ├── ensembles.py        # Hard Voting, Calibrated Soft Voting, Stacking (Meta-LR)
│   ├── evaluation.py       # Macro F1, MCC, AUPRC, Accuracy, error analysis
│   ├── visualization.py    # Publication plots (EDA, Confusion Matrix, ROC, PR, Importances)
│   └── utils.py            # Timers, deterministic seeding, serialization helpers
│
├── tests/
│   ├── test_data.py        # Dataset loading, cleaning, and split integrity tests
│   ├── test_features.py    # Feature extraction and dimensionality tests
│   └── test_models.py      # Baseline estimators, Platt calibration, and ensemble mechanics
│
├── results/
│   ├── figures/            # 23 generated publication-quality plots (300 DPI)
│   ├── metrics/            # Tabular metric reports (.csv, .json)
│   └── predictions/        # Test-set predictions (.npy)
│
├── models/
│   └── saved/              # Serialized checkpoints and preprocessors (.pkl, .json)
│
├── reports/
│   ├── figures/            # Executive summary report figures
│   └── final_report.md     # In-depth biological error analysis report
│
├── run_pipeline.py         # Headless CLI pipeline runner
├── requirements.txt        # Pinned runtime dependencies
├── .gitignore              # Configured for GitHub safety (<100MB policy)
└── README.md
```

---

## ⚙️ Installation & Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/Tesfalegnp/Enzyme-Function-Classification.git
cd Enzyme-Function-Classification
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage

### Option A: Presentation Notebook (Recommended)
Launch Jupyter Lab to explore the preserved 40-cell presentation artifact:
```bash
jupyter lab notebooks/enzyme_function_classification.ipynb
```
Every cell from **Cell 1 through Cell 40** contains detailed markdown documentation (Objective, Why, Expected Output) and renders all figures, confusion matrices, ROC/PR curves, and comparison tables.

### Option B: Headless CLI Pipeline Runner
Execute the modular pipeline stages from the command line:
```bash
# 1. Run Phase 1 diagnostics and verification checks
python3 run_pipeline.py --stage verify

# 2. Run Phase 2 cleaning, deduplication, and 70/15/15 stratified split
python3 run_pipeline.py --stage prep

# 3. Run Phase 3 EDA and Phase 4 feature extraction
python3 run_pipeline.py --stage features
```

### Option C: Automated Test Suite
Run the 13 automated unit tests:
```bash
python3 -m pytest tests/ -v
```

---

## 🛡️ Scientific Rigor & Leakage Prevention Controls

1. **Zero Test-Set Peeking**: The test partition ($31,841$ samples) remained completely untouched until **Cell 33**. No test-set statistics, distributions, or labels were accessed during feature selection, model training, calibration, or threshold tuning.
2. **Leakage-Safe Feature Selection**: `SelectKBest` (ANOVA F-statistic, $k=500$) was fitted strictly on training data; validation and test splits were transformed using only training statistics.
3. **Platt-Calibrated SVM Probabilities**: `LinearSVC.decision_function()` outputs raw signed margin distances, which are not probabilities. Calibrated posterior probabilities $P(y=c|x)$ were derived via Platt scaling (`CalibratedClassifierCV`) fitted strictly within training data.
4. **Leakage-Free Stacking**: Meta-features for training the Logistic Regression meta-learner were constructed via 3-fold stratified out-of-fold (OOF) cross-validation on `X_train_final`.
5. **Deterministic Reproducibility**: All random seeds across NumPy, Scikit-Learn, and LightGBM were explicitly pinned to `42`.

---

## 📈 Visualizations & Artifact Highlights

| Visualization | Description | File Link |
| :--- | :--- | :--- |
| **EDA Distributions** | EC class distribution and sequence length across splits | [`results/figures/eda_class_and_length_distribution.png`](results/figures/eda_class_and_length_distribution.png) |
| **Normalized Confusion Matrix** | Stacking classifier normalized true-class error matrix | [`results/figures/confusion_matrix_stacking.png`](results/figures/confusion_matrix_stacking.png) |
| **One-vs-Rest ROC Curves** | Multiclass ROC curves across all 6 models ($0.9622$ Macro AUC) | [`results/figures/roc_curves_all_models.png`](results/figures/roc_curves_all_models.png) |
| **Precision-Recall Curves** | Multiclass PR curves across all 6 models ($0.9016$ Macro AUPRC) | [`results/figures/pr_curves_all_models.png`](results/figures/pr_curves_all_models.png) |
| **Feature Importances** | Top 20 ranked biological features for RF and LightGBM | [`results/figures/feature_importance_comparison.png`](results/figures/feature_importance_comparison.png) |
| **Sequence Length Error Rate** | Classification error rates across length bins ($<200$ to $>1200$ aa) | [`results/figures/error_rate_by_sequence_length.png`](results/figures/error_rate_by_sequence_length.png) |

---

## 💡 Key Biological & Technical Insights

1. **Algorithmic Complementarity**: Random Forest and LinearSVC exhibit a `36.21%` prediction divergence on test cases. Stacking successfully leverages these complementary inductive biases (tree-based non-linear partitioning vs. maximum-margin linear hyperplanes) to achieve superior test accuracy ($79.77\%$).
2. **Dominant Feature Motifs**: The leading principal component of the 8,000-dimensional tripeptide space (`SVD_dim_0`) and individual amino-acid fractions (`AAC_C` for disulfide stability, `AAC_G` for backbone flexibility) proved to be the most influential predictive markers.
3. **Sequence Length Association**: Sequences shorter than 200 amino acids showed an elevated error rate ($24.6\%$), reflecting challenges in classifying truncated catalytic domains without full contextual motifs.

---

## 📜 License & Citation

This project is licensed under the **MIT License**.

If you utilize this pipeline, codebase, or methodology in your biological machine learning research, please cite:

```bibtex
@misc{enzyme_function_classification_2026,
  author = {Tesfalegn, P.},
  title = {Enzyme Function Classification (EC 1–6) using Machine Learning},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Tesfalegnp/Enzyme-Function-Classification}}
}
```
