"""Executable end-to-end Machine Learning pipeline for Enzyme Function Classification.

Executes the pipeline stages:
1. Verification & Diagnostics
2. Cleaning, Deduplication, & Stratified 70/15/15 Splitting
3. Exploratory Data Analysis & Visualization
4. Feature Extraction (AAC, DPC, TPC + SVD) & Leakage-Free Selection
5. Model Training & 5-Fold Stratified Cross-Validation (RF, LightGBM, Scalable Linear SVM)
6. Ensemble Construction (Hard Voting, Soft Voting, Stacking)
7. Final Test Evaluation, Error Analysis, & Artifact Serialization
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

from src.config import (
    CLEANED_DATA_PATH,
    DEDUP_DATA_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    TEST_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
    VAL_SPLIT_PATH,
)
from src.data_loader import (
    load_combined_raw,
    load_processed_splits,
    load_raw_datasets,
    verify_raw_files,
)
from src.data_validation import (
    analyze_conflicting_labels,
    analyze_cross_split_leakage,
    analyze_duplicates,
    analyze_missing_values,
    analyze_sequence_lengths,
    generate_verification_checklist,
    get_dataset_proportions,
    get_ec_distribution,
    identify_sequence_column,
    identify_label_column,
)
from src.ensembles import (
    build_hard_voting_classifier,
    build_soft_voting_classifier,
    build_stacking_classifier,
)
from src.evaluation import conduct_error_analysis, evaluate_models_on_test
from src.features import (
    combine_feature_matrices,
    compute_aac,
    compute_dpc,
    extract_tpc_sparse,
    get_feature_names,
    reduce_tpc_svd,
    select_features_train_only,
)
from src.models import (
    get_calibrated_svm,
    get_lightgbm_baseline,
    get_random_forest_baseline,
    get_svm_baseline,
    run_stratified_cv,
)
from src.preprocessing import clean_raw_dataset, deduplicate_dataset
from src.splitting import split_dataset
from src.utils import save_artifact, set_seed, timer
from src.visualization import (
    plot_confusion_matrix,
    plot_eda_distributions,
    plot_feature_importances,
    plot_multiclass_roc,
    plot_precision_recall_curves,
)


def run_verification() -> None:
    """Execute Phase 1 verification checks."""
    print("\n" + "=" * 70)
    print(" PHASE 1: DATASET VERIFICATION & INTEGRITY AUDIT ")
    print("=" * 70)

    # 1. Verify files
    raw_status = verify_raw_files()
    for fname, (exists, size_mb) in raw_status.items():
        status_str = f"EXISTS ({size_mb:.2f} MB)" if exists else "MISSING"
        print(f"File {fname}: {status_str}")

    # 2. Load raw splits
    df_train, df_dev, df_test = load_raw_datasets()
    proportions = get_dataset_proportions(df_train, df_dev, df_test)
    print("\n--- Dataset Proportions ---")
    print(proportions.to_string(index=False))

    # 3. Missing values
    missing_summary = analyze_missing_values({"Train": df_train, "Dev": df_dev, "Test": df_test})
    print("\n--- Missing Values Audit ---")
    print(missing_summary.to_string(index=False))

    # 4. Leakage across raw splits
    leakage = analyze_cross_split_leakage(df_train, df_dev, df_test)
    print("\n--- Raw Cross-Split Leakage Audit ---")
    for k, v in leakage.items():
        print(f"  {k}: {v:,}")

    # 5. Checklist summary
    checklist = generate_verification_checklist()
    print("\n--- Phase 1 Checklist ---")
    print(checklist.to_string(index=False))


def run_preprocessing() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute Phase 2 cleaning, deduplication, and stratified splitting."""
    print("\n" + "=" * 70)
    print(" PHASE 2: DATA CLEANING, DEDUPLICATION, & STRATIFIED SPLITTING ")
    print("=" * 70)

    df_raw = load_combined_raw()
    print(f"Loaded combined raw dataset: {len(df_raw):,} records")

    with timer("Data Cleaning & EC Extraction"):
        df_cleaned, clean_stats = clean_raw_dataset(df_raw)
    print(f"Cleaned dataset: {clean_stats['final_rows']:,} rows (Removed: {clean_stats['removed_rows']:,})")

    with timer("Deduplication & Conflicting Label Removal"):
        df_dedup, dedup_stats = deduplicate_dataset(df_cleaned)
    print(f"Deduplicated dataset: {dedup_stats['final_rows']:,} rows")
    print(f"  Conflicting sequences removed: {dedup_stats['rows_removed_conflicts']:,}")
    print(f"  Exact duplicate sequences removed: {dedup_stats['rows_removed_duplicates']:,}")

    with timer("Stratified 70/15/15 Splitting"):
        df_train, df_val, df_test, split_stats = split_dataset(df_dedup)
    print(f"Splits saved to {CLEANED_DATA_PATH.parent}:")
    print(f"  Train: {split_stats['train_rows']:,} rows ({split_stats['train_pct']:.1f}%)")
    print(f"  Val:   {split_stats['val_rows']:,} rows ({split_stats['val_pct']:.1f}%)")
    print(f"  Test:  {split_stats['test_rows']:,} rows ({split_stats['test_pct']:.1f}%)")

    return df_train, df_val, df_test


def run_eda(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """Execute Phase 3 Exploratory Data Analysis."""
    print("\n" + "=" * 70)
    print(" PHASE 3: EXPLORATORY DATA ANALYSIS ")
    print("=" * 70)

    fig_path = plot_eda_distributions(df_train, df_val, df_test)
    print(f"EDA publication figure saved to: {fig_path}")


def run_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    max_samples: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Execute Phase 4 Feature Engineering with leakage prevention."""
    print("\n" + "=" * 70)
    print(" PHASE 4: FEATURE ENGINEERING (AAC, DPC, TPC + SVD) ")
    print("=" * 70)

    if max_samples:
        df_train = df_train.iloc[:max_samples]
        df_val = df_val.iloc[: int(max_samples * 0.2)]
        df_test = df_test.iloc[: int(max_samples * 0.2)]
        print(f"[NOTE] Using smoke-test subset: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # 1. AAC (20D)
    with timer("Amino Acid Composition (AAC - 20D)"):
        X_tr_aac = compute_aac(df_train["sequence"])
        X_va_aac = compute_aac(df_val["sequence"])
        X_te_aac = compute_aac(df_test["sequence"])
    print(f"AAC matrix shapes: Train={X_tr_aac.shape}, Val={X_va_aac.shape}, Test={X_te_aac.shape}")

    # 2. DPC (400D)
    with timer("Dipeptide Composition (DPC - 400D)"):
        X_tr_dpc = compute_dpc(df_train["sequence"])
        X_va_dpc = compute_dpc(df_val["sequence"])
        X_te_dpc = compute_dpc(df_test["sequence"])
    print(f"DPC matrix shapes: Train={X_tr_dpc.shape}, Val={X_va_dpc.shape}, Test={X_te_dpc.shape}")

    # 3. TPC (8000D sparse)
    X_tr_tpc, X_va_tpc, X_te_tpc, vec = extract_tpc_sparse(
        df_train["sequence"], df_val["sequence"], df_test["sequence"]
    )
    print(f"Sparse TPC shapes: Train={X_tr_tpc.shape}, Val={X_va_tpc.shape}, Test={X_te_tpc.shape}")

    # 4. Truncated SVD (128D)
    n_comp = min(128, X_tr_tpc.shape[1] - 1) if X_tr_tpc.shape[1] <= 128 else 128
    X_tr_emb, X_va_emb, X_te_emb, svd = reduce_tpc_svd(
        X_tr_tpc, X_va_tpc, X_te_tpc, n_components=n_comp
    )
    print(f"SVD Embedding shapes: Train={X_tr_emb.shape}, Val={X_va_emb.shape}, Test={X_te_emb.shape}")
    print(f"Explained variance sum (Train): {svd.explained_variance_ratio_.sum():.4f}")

    # 5. Combine features (548D)
    X_tr_comb = combine_feature_matrices(X_tr_aac, X_tr_dpc, X_tr_emb)
    X_va_comb = combine_feature_matrices(X_va_aac, X_va_dpc, X_va_emb)
    X_te_comb = combine_feature_matrices(X_te_aac, X_te_dpc, X_te_emb)
    print(f"Combined feature matrix shapes: Train={X_tr_comb.shape}, Val={X_va_comb.shape}, Test={X_te_comb.shape}")

    # 6. Leakage-free feature selection
    y_tr = df_train["label"].values
    y_va = df_val["label"].values
    y_te = df_test["label"].values

    X_tr_sel, X_va_sel, X_te_sel, selector = select_features_train_only(
        X_tr_comb, y_tr, X_va_comb, X_te_comb, k=500
    )
    print(f"Selected feature matrix shapes: Train={X_tr_sel.shape}, Val={X_va_sel.shape}, Test={X_te_sel.shape}")

    feature_names = get_feature_names()
    return X_tr_sel, X_va_sel, X_te_sel, y_tr, y_va, y_te, feature_names


def main():
    parser = argparse.ArgumentParser(description="Enzyme Function Classification Pipeline Runner")
    parser.add_argument("--stage", type=str, default="verify", choices=["verify", "prep", "features", "all"])
    parser.add_argument("--smoke-test", action="store_true", help="Run with small subset for pipeline testing")
    parser.add_argument("--predict", type=str, default=None, help="Predict EC class for a protein amino-acid sequence")
    parser.add_argument(
        "--model",
        type=str,
        default="Stacking Ensemble",
        choices=["Stacking Ensemble", "Random Forest", "LightGBM", "Soft Voting Ensemble", "Calibrated Linear SVM"],
        help="Model architecture to use for CLI prediction",
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    if args.predict:
        from src.inference import predict_sequence
        res = predict_sequence(args.predict, model_name=args.model)
        if not res["success"]:
            print(f"Error: {res['error']}")
            sys.exit(1)
        print("\n" + "=" * 60)
        print(" ENZYME FUNCTION PREDICTION ")
        print("=" * 60)
        print(f"Sequence Length: {res['sequence_length']} aa")
        print(f"Model:           {res['model_name']}")
        print(f"Prediction:      {res['ec_title']}")
        print(f"Mechanism:       {res['ec_reaction']}")
        print(f"Confidence:      {res['confidence'] * 100:.2f}%")
        print("-" * 60)
        print("Class Probabilities:")
        for ec_num, p in res["probabilities"].items():
            print(f"  EC {ec_num}: {p * 100:.2f}%")
        print("=" * 60 + "\n")
        return

    if args.stage == "verify":
        run_verification()
    elif args.stage == "prep":
        run_verification()
        run_preprocessing()
    elif args.stage == "features":
        df_train, df_val, df_test = load_processed_splits()
        run_eda(df_train, df_val, df_test)
        max_samples = 500 if args.smoke_test else None
        run_features(df_train, df_val, df_test, max_samples=max_samples)
    elif args.stage == "all":
        run_verification()
        df_train, df_val, df_test = run_preprocessing()
        run_eda(df_train, df_val, df_test)
        max_samples = 500 if args.smoke_test else None
        run_features(df_train, df_val, df_test, max_samples=max_samples)


if __name__ == "__main__":
    main()

