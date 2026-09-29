# Enzyme Function Classification from Protein Sequences

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/tests-23%2F23%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, scientifically validated machine learning system for predicting primary enzyme catalytic functions (**EC 1 through EC 6**) directly from protein primary amino-acid sequences using the SwissProt-EC benchmark ($212,273$ curated sequences).

Featuring leakage-free multi-scale feature engineering, scalable baseline models, Platt calibration, stacking ensemble learning, automated unit tests, and an interactive **Streamlit web application** for real-time inference.

---

## 🔬 1. Project Overview

Enzymes are biological catalysts essential to all metabolic and biochemical pathways. To catalog enzyme catalyzed reactions systematically, the International Union of Biochemistry and Molecular Biology established the **Enzyme Commission (EC)** classification system.

This project addresses the challenge of **predicting the primary (top-level) EC classification** of an uncharacterized protein directly from its primary amino-acid sequence.

- **Input**: Raw protein primary amino-acid sequence (plain string or standard multi-line FASTA format).
- **Output**: Top-level Enzyme Commission category (**EC 1 through EC 6**) with calibrated class posterior probabilities.
- **Scope Note**: This system specifically predicts the **top-level (first-digit) primary enzyme category**, rather than the full four-level hierarchical notation (e.g., predicting `EC 1` for an oxidoreductase rather than `EC 1.1.1.1`).

---

## 🏷️ 2. Primary Enzyme Classes (EC 1–6)

| Class | Official Category | Catalyzed Chemical Reaction | Representative Enzyme Types |
| :---: | :--- | :--- | :--- |
| **EC 1** | **Oxidoreductases** | Oxidation-reduction reactions (electron, hydride, or hydrogen transfer) | Dehydrogenases, oxidases, reductases, peroxidases |
| **EC 2** | **Transferases** | Transfer of functional chemical groups (methyl, acyl, phospho, glycosyl) | Kinases, transaminases, methyltransferases, polymerases |
| **EC 3** | **Hydrolases** | Hydrolytic cleavage of chemical bonds (C-O, C-N, C-C) utilizing water | Proteases, nucleases, esterases, phosphatases, lipases |
| **EC 4** | **Lyases** | Non-hydrolytic and non-oxidative elimination forming double bonds or rings | Decarboxylases, aldolases, synthases, dehydratases |
| **EC 5** | **Isomerases** | Geometric, structural, or spatial rearrangements within a single molecule | Racemases, epimerases, mutases, topoisomerases |
| **EC 6** | **Ligases** | Joining of two molecules coupled with pyrophosphate bond cleavage (ATP) | Synthetases, DNA/RNA ligases, carboxylases |

---

## 📁 3. Dataset & Data Curation

The project trains and evaluates on the curated **SwissProt-EC benchmark dataset**, derived from UniProtKB/Swiss-Prot high-confidence annotations.

### Dataset Pipeline & Filtering Statistics

1. **Raw Combined Ingestion**:
   - `train-00000-of-00001.parquet`: $208,823$ records
   - `dev-00000-of-00001.parquet`: $26,724$ records
   - `test-00000-of-00001.parquet`: $25,892$ records
   - Combined Raw Volume: **$221,489$ records**
2. **Data Cleaning & Filtering**:
   - Extracted primary EC digit (1–6).
   - Removed records with missing sequences, null labels, or invalid EC numbers ($316$ rows removed).
   - Cleaned volume: $221,173$ records.
3. **Conflict Resolution & Deduplication**:
   - Removed **$88$ sequences** with conflicting annotations (identical sequence mapped to multiple distinct primary EC classes, removing $177$ conflicting rows).
   - Removed **$8,723$ exact duplicate sequences** (keeping first occurrence).
   - Final Curated Dataset: **$212,273$ unique sequence-label pairs**.
4. **Stratified 70/15/15 Partitioning**:
   - **Training Set (70.0%)**: $148,591$ samples
   - **Validation Set (15.0%)**: $31,841$ samples
   - **Held-Out Test Set (15.0%)**: $31,841$ samples
   - **Zero Cross-Split Leakage**: Confirmed 0 sequence overlap across partitions.

### Class Distribution (Curated Dataset)

```text
EC 1 — Oxidoreductases :  27,788 (13.09%)
EC 2 — Transferases     :  78,597 (37.03%)
EC 3 — Hydrolases       :  50,393 (23.74%)
EC 4 — Lyases           :  19,622 ( 9.24%)
EC 5 — Isomerases       :  11,392 ( 5.37%)
EC 6 — Ligases          :  24,481 (11.53%)
-----------------------------------------
Total Unique Sequences  : 212,273 (100.0%)
```

---

## 🧬 4. Methodology & Pipeline Architecture

```text
Raw SwissProt-EC Parquet Files
               ↓
    Phase 1: Verification & Integrity Audit (zero leakage check)
               ↓
    Phase 2: Cleaning, Deduplication & Conflict Pruning
               ↓
    Stratified Partitioning (70% Train / 15% Val / 15% Test)
               ↓
┌──────────────────────────────────────────────────────────────┐
│  Phase 4: Multi-Scale Biological Feature Extraction          │
│  ├─ AAC: Amino Acid Composition (20D normalized)             │
│  ├─ DPC: Dipeptide Composition (400D normalized)             │
│  └─ TPC: Tripeptide (8,000D sparse) → TruncatedSVD (128D)    │
│  Combined Representation: 548 Features                       │
│  Feature Selection: SelectKBest (ANOVA F, k=500, Train-Only) │
└──────────────────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────┐
│  Phase 5 & 6: Machine Learning Models & Ensembles            │
│  ├─ Random Forest (100 trees, depth 20, class-balanced)      │
│  ├─ LightGBM (100 trees, 31 leaves, feature fraction 0.5)    │
│  ├─ LinearSVC (C=0.05) + Platt Scaling Calibration           │
│  ├─ Hard Voting (Majority rule, RF priority tie-breaker)     │
│  ├─ Soft Voting (Equal-weighted calibrated posterior mean)   │
│  └─ Stacking Classifier (3-Fold Stratified OOF Meta-Learner) │
└──────────────────────────────────────────────────────────────┘
               ↓
    Phase 7: Held-Out Test Evaluation & Biological Error Analysis
               ↓
    Real-Time Interactive Inference UI (Streamlit)
```

### Leakage Prevention Guarantees

- **Training-Only Transformers**: The `CountVectorizer` (tripeptides), `TruncatedSVD` (128D), `SelectKBest` ($k=500$), and `StandardScaler` were fit strictly on `X_train`. Validation and test splits were transformed using frozen training parameters.
- **Strictly Untouched Test Set**: The $31,841$ test sequences were never accessed during feature selection, model training, hyperparameter tuning, or stacking meta-learner training.
- **Out-of-Fold (OOF) Stacking**: Meta-features for training the Stacking meta-learner (`LogisticRegression`) were generated via 3-fold stratified cross-validation strictly on the training partition.

---

## 📊 5. Benchmark Results

Evaluated across **Macro F1** (primary decision metric), **Accuracy**, **Matthews Correlation Coefficient (MCC)**, and **Macro AUPRC**.

### Comparison: Validation vs. Final Held-Out Test Performance

| Model / Ensemble Architecture | Accuracy (Val) | Macro F1 (Val) | MCC (Val) | Accuracy (Test) | Macro F1 (Test) | MCC (Test) | Macro AUPRC (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (100 trees, depth 20) | 0.7858 | 0.7939 | 0.7189 | 0.7884 | **0.7945** | 0.7222 | 0.8713 |
| **LightGBM** (100 trees, 31 leaves) | 0.7378 | 0.7420 | 0.6666 | 0.7395 | 0.7428 | 0.6688 | 0.8442 |
| **LinearSVC** ($C=0.05$, Platt-calibrated) | 0.5575 | 0.5351 | 0.4399 | 0.5565 | 0.5359 | 0.4402 | 0.5916 |
| **Hard Voting Ensemble** (Majority rule) | 0.7729 | 0.7766 | 0.7044 | 0.7744 | 0.7776 | 0.7065 | N/A* |
| **Soft Voting Ensemble** (Equal weights) | 0.7543 | 0.7599 | 0.6747 | 0.7587 | 0.7645 | 0.6807 | 0.8448 |
| **Stacking Classifier** (Meta-Learner)** | **0.7926** | **0.7905** | **0.7362** | **0.7977** | 0.7942 | **0.7426** | **0.9016** |

*\*Hard voting produces discrete categorical votes; continuous probability rankings are not defined.*  
*\*\*Stacking Classifier achieved the highest Test Accuracy, Test MCC, and Test Macro AUPRC.*

### Per-Class Test Performance (Stacking Ensemble on $31,841$ Test Samples)

| Class | Category | Precision | Recall | F1-Score | Support | ROC-AUC | AUPRC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EC 1** | Oxidoreductases | 0.8037 | 0.7711 | 0.7871 | 4,168 | 0.9687 | 0.8994 |
| **EC 2** | Transferases | 0.7601 | 0.8540 | 0.8043 | 11,790 | 0.9381 | 0.9023 |
| **EC 3** | Hydrolases | 0.8354 | 0.8491 | 0.8422 | 7,559 | 0.9427 | 0.8931 |
| **EC 4** | Lyases | 0.7699 | 0.6337 | 0.6952 | 2,943 | 0.9458 | 0.7423 |
| **EC 5** | Isomerases | 0.8148 | 0.7718 | 0.7927 | 1,709 | 0.9879 | 0.9575 |
| **EC 6** | Ligases | 0.8876 | 0.8848 | 0.8862 | 3,672 | 0.9897 | 0.9576 |
| **Macro Avg** | — | **0.8119** | **0.7941** | **0.7942** | **31,841** | **0.9622** | **0.9016** |

---

## 🏆 6. Final Model Selection Rationale

- **Highest Test Accuracy & AUPRC**: The **Stacking Classifier** achieves the highest test overall accuracy (**$79.77\%$**), highest MCC (**$0.7426$**), and highest Macro AUPRC (**$0.9016$**).
- **Highest Macro F1**: **Random Forest** achieved the highest test Macro-F1 (**$0.7945$** vs. Stacking's $0.7942$).
- **Complementarity**: The base learners exhibit distinct inductive biases. Tree-based partitioning in Random Forest and LightGBM captures non-linear k-mer interactions, while linear margin separation in LinearSVC provides orthogonal boundary support. The stacking meta-learner successfully exploits these differences without overfitting.

---

## 📈 7. Results Visualizations

Selected key figures generated during validation and test evaluation:

| Visualization | Description | File Path |
| :--- | :--- | :--- |
| **Dataset Distribution** | Class counts and sequence length distributions across partitions | [`results/figures/eda_class_and_length_distribution.png`](results/figures/eda_class_and_length_distribution.png) |
| **Stacking Confusion Matrix** | Normalized true-vs-predicted confusion matrix on held-out test data | [`results/figures/confusion_matrix_stacking.png`](results/figures/confusion_matrix_stacking.png) |
| **Multiclass ROC Curves** | One-vs-Rest ROC curves across all 6 models ($0.9622$ Macro AUC) | [`results/figures/roc_curves_all_models.png`](results/figures/roc_curves_all_models.png) |
| **Precision-Recall Curves** | One-vs-Rest PR curves across all 6 models ($0.9016$ Macro AUPRC) | [`results/figures/pr_curves_all_models.png`](results/figures/pr_curves_all_models.png) |
| **Feature Importances** | Top ranked biological features (AAC, DPC, and SVD components) | [`results/figures/feature_importance_comparison.png`](results/figures/feature_importance_comparison.png) |
| **Sequence Length Error Rate** | Error rates stratified across amino-acid length bins ($<200$ to $>1200$ aa) | [`results/figures/error_rate_by_sequence_length.png`](results/figures/error_rate_by_sequence_length.png) |

---

## 💻 8. Interactive Web UI (Streamlit)

A scientific single-page web application is provided for real-time testing and demonstration.

### Features
- **Flexible Sequence Input**: Supports single-line, multi-line, and standard FASTA formatted sequences.
- **Input Validation**: Strips whitespace, numbering, and FASTA headers; validates against the standard 20 amino acids; flags non-standard characters with clear feedback.
- **One-Click Verified Examples**: Load authentic SwissProt-EC test partition samples with known EC labels.
- **Zero-Training Inference**: Preloads trained models and transformers once from disk via `@st.cache_resource`.
- **Comprehensive Diagnostic Output**: Displays predicted EC class, catalyzed chemical reaction mechanism, posterior class probability bars, base model predictions, and amino-acid composition.

### Launching the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🛠️ 9. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10–3.14 on Linux/macOS/Windows)
- Recommended: virtual environment (`venv` or `conda`)

### Step-by-Step Setup
```bash
# 1. Clone the repository
git clone https://github.com/Tesfalegnp/Enzyme-Function-Classification.git
cd Enzyme-Function-Classification

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚡ 10. Usage & Execution Commands

### 1. Run the Interactive Web UI
```bash
streamlit run app.py
```

### 2. Run the Automated Test Suite (23 Unit Tests)
```bash
pytest -v
```

### 3. Run Command-Line Single-Sequence Inference
```bash
# Run CLI prediction using the stacking ensemble:
python3 run_pipeline.py --predict "MRRKIVVGNWKMNNSVAESVQLATDVLAALGEGFSGCEVGIAPTYLALDATEKVIAESEVQLVAQNCHYENDGAFTGEVSARMILAVGCSSVIIGHSERRQYFGETNATVNLRIKKALSEGLNVILCVGETLAERESGVMETVISSQVREGLDGIIDISAIVIAYEPVWAIGTGKTASSAQAEEVHLFIRTLVTGLYGQTASEKVRIQYGGSVKPSNAAELFAMPNIDGGLIGGASLNADDFAAIVKAASV"
```

### 4. Run Pipeline Stages
```bash
# Run Phase 1 dataset diagnostics and zero-leakage verification:
python3 run_pipeline.py --stage verify

# Run Phase 2 cleaning, deduplication, and stratified 70/15/15 splitting:
python3 run_pipeline.py --stage prep

# Run Phase 3 EDA and Phase 4 feature extraction:
python3 run_pipeline.py --stage features
```

### 5. Explore the Presentation Notebook
```bash
jupyter lab notebooks/enzyme_function_classification.ipynb
```

---

## 📂 11. Project Directory Structure

```text
Enzyme_Classification)_in_ML/
├── app.py                     # Interactive Streamlit Web UI application
├── run_pipeline.py            # CLI pipeline runner with --stage and --predict
├── pytest.ini                 # Pytest configuration with automatic pythonpath
├── requirements.txt           # Pinned runtime and UI dependencies
├── LICENSE                    # MIT License
├── README.md                  # Comprehensive project documentation
│
├── src/
│   ├── __init__.py            # Package declaration (v1.0.0)
│   ├── config.py              # Centralized paths, seed (42), dimensions, constants
│   ├── data_loader.py         # Raw parquet and split loading utilities
│   ├── data_validation.py     # Phase 1 diagnostic and cross-split leakage checks
│   ├── preprocessing.py       # Sequence cleaning, EC extraction, deduplication
│   ├── splitting.py           # Stratified 70/15/15 partitioning (train/val/test)
│   ├── features.py            # AAC (20D), DPC (400D), TPC (8000D), SVD (128D), SelectKBest
│   ├── models.py              # RF, LightGBM, LinearSVC, 5-fold CV, tuning
│   ├── ensembles.py           # Hard Voting, Soft Voting, Stacking (Meta-LR)
│   ├── evaluation.py          # Macro F1, MCC, AUPRC, Accuracy, error analysis
│   ├── visualization.py       # Publication plots (EDA, Confusion Matrix, ROC, PR)
│   ├── inference.py           # Reusable, zero-training inference engine
│   └── utils.py               # Timers, deterministic seeding, serialization helpers
│
├── tests/
│   ├── test_data.py           # 5 tests: data cleaning, deduplication, splitting
│   ├── test_features.py       # 4 tests: AAC, DPC, TPC + SVD, SelectKBest
│   ├── test_models.py         # 4 tests: baselines, calibration, ensembles, metrics
│   └── test_inference.py      # 10 tests: validation, feature extraction, predictions
│
├── models/
│   └── saved/                 # Serialized model checkpoints & preprocessors
│       ├── calibrated_svm.pkl
│       ├── lightgbm.pkl
│       ├── random_forest.pkl
│       ├── select_k_best.pkl
│       ├── stacking_meta_learner.pkl
│       ├── standard_scaler.pkl
│       ├── svm.pkl
│       ├── tripeptide_vectorizer.pkl
│       ├── truncated_svd.pkl
│       ├── base_model_val_metrics.json
│       └── ensemble_metadata.json
│
├── results/
│   ├── figures/               # 23 generated publication figures (300 DPI)
│   ├── metrics/               # Evaluation CSV & JSON metric summaries
│   └── predictions/           # Test set prediction arrays (.npy)
│
├── data/
│   ├── raw/swissprot-ec/      # Raw SwissProt-EC parquet files
│   └── processed/             # Cleaned, deduplicated splits and feature matrices
│
└── notebooks/
    └── enzyme_function_classification.ipynb  # Primary 40-cell presentation notebook
```

---

## ⚠️ 12. Scientific Limitations

1. **Top-Level EC Prediction Only**: This system predicts the primary enzyme category (EC 1–6). It does not predict sub-classes, sub-sub-classes, or catalytic serial numbers (e.g., predicting EC 1 rather than EC 1.1.1.1).
2. **Primary Sequence Dependency**: Predictions are made solely from 1D primary amino-acid composition and short k-mer frequencies; 3D tertiary conformations, active site pockets, cofactor binding, and quaternary structures are not explicitly modeled.
3. **Random Stratified Partitioning**: The dataset is split using stratified random sampling rather than strict sequence-identity clustering (e.g., MMseqs2 or CD-HIT 30% sequence identity cutoffs). Homologous proteins between train and test may lead to higher apparent generalization scores compared to strict cluster splits.
4. **Calibrated Posteriors**: Reported class probabilities are outputs of Platt scaling and the logistic regression meta-learner, reflecting relative empirical confidence rather than physical thermodynamic affinities.
5. **Experimental Verification**: In silico predictions must serve as hypothesis generators and cannot replace wet-lab enzyme assays or crystallographic validation.

---

## 🔮 13. Future Directions

- **Homology-Aware Evaluation**: Incorporate MMseqs2/CD-HIT sequence-identity clustering splits (e.g. 30% cutoff) to measure zero-shot generalization to novel protein families.
- **Deep Sequence Embeddings**: Benchmark pre-trained biological protein language models (ESM-2, ProtBERT, Ankh) against classical k-mer compositions.
- **Hierarchical Classification**: Extend the architecture from top-level classification to full 4-digit hierarchical EC prediction via tree-structured classifiers or multi-task neural networks.
- **Structural Integration**: Incorporate predicted 3D contact maps or AlphaFold-derived structural descriptors.

---

## 📜 14. License & Citation

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

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
