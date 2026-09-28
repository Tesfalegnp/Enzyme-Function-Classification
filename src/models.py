"""Model definitions, scalable cross-validation harnesses, and hyperparameter tuning."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import CV_FOLDS, N_JOBS, RANDOM_SEED, SELECT_K_FEATURES
from src.utils import timer


def get_random_forest_baseline(
    n_estimators: int = 100,
    max_depth: Optional[int] = 20,
    max_features: Union[str, float] = "sqrt",
    random_state: int = RANDOM_SEED,
    n_jobs: int = N_JOBS,
) -> RandomForestClassifier:
    """Instantiate baseline Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def get_lightgbm_baseline(
    n_estimators: int = 100,
    num_leaves: int = 31,
    colsample_bytree: float = 0.5,
    subsample: float = 0.8,
    subsample_freq: int = 1,
    random_state: int = RANDOM_SEED,
    n_jobs: int = N_JOBS,
) -> lgb.LGBMClassifier:
    """Instantiate baseline LightGBM classifier with feature subsampling."""
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        subsample_freq=subsample_freq,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
    )


def get_svm_baseline(
    C: float = 0.05,
    tol: float = 1e-3,
    max_iter: int = 2000,
    random_state: int = RANDOM_SEED,
) -> LinearSVC:
    """Instantiate scalable Linear Support Vector Classifier."""
    return LinearSVC(
        C=C,
        class_weight="balanced",
        dual=False,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
    )


def get_calibrated_svm(
    base_svm: Optional[LinearSVC] = None,
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Wrap LinearSVC in Platt scaling (CalibratedClassifierCV) for true posterior probabilities."""
    if base_svm is None:
        base_svm = get_svm_baseline()
    return CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=cv,
        n_jobs=N_JOBS,
    )


def run_stratified_cv(
    model: Any,
    X: np.ndarray,
    y: Union[np.ndarray, pd.Series],
    model_name: str = "Model",
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_SEED,
    apply_feature_selection_in_fold: bool = False,
    k_features: int = SELECT_K_FEATURES,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute stratified 5-fold cross-validation with per-fold metrics and timing.

    If apply_feature_selection_in_fold is True, feature selection is fitted strictly
    inside each training fold to eliminate any potential data leakage across folds.
    If scaler is provided, it is fitted strictly on the fold training data.
    """
    y_arr = np.array(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_scores: List[float] = []
    fold_accuracies: List[float] = []
    fold_mccs: List[float] = []
    fold_times: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_arr), 1):
        print(f"[START] Fold {fold}/{n_splits}...")
        t0 = time.perf_counter()
        X_tr, y_tr = X[train_idx], y_arr[train_idx]
        X_va, y_val = X[val_idx], y_arr[val_idx]

        if apply_feature_selection_in_fold:
            selector = SelectKBest(score_func=f_classif, k=min(k_features, X.shape[1]))
            X_tr = selector.fit_transform(X_tr, y_tr)
            X_va = selector.transform(X_va)

        if scaler is not None:
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        elapsed = time.perf_counter() - t0

        macro_f1 = float(f1_score(y_val, preds, average="macro"))
        acc = float(accuracy_score(y_val, preds))
        mcc = float(matthews_corrcoef(y_val, preds))

        fold_scores.append(macro_f1)
        fold_accuracies.append(acc)
        fold_mccs.append(mcc)
        fold_times.append(elapsed)

        print(f"[COMPLETE] Fold {fold}/{n_splits}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  MCC:      {mcc:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Time:     {elapsed:.2f} sec\n")

    scores_arr = np.array(fold_scores)
    acc_arr = np.array(fold_accuracies)
    mcc_arr = np.array(fold_mccs)
    total_time = float(sum(fold_times))

    results = {
        "model_name": model_name,
        "fold_macro_f1": fold_scores,
        "mean_macro_f1": float(scores_arr.mean()),
        "std_macro_f1": float(scores_arr.std()),
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(acc_arr.mean()),
        "std_accuracy": float(acc_arr.std()),
        "fold_mccs": fold_mccs,
        "mean_mcc": float(mcc_arr.mean()),
        "std_mcc": float(mcc_arr.std()),
        "total_time_seconds": total_time,
        "mean_time_per_fold": float(np.mean(fold_times)),
    }

    print("-" * 50)
    print("CV SUMMARY")
    print(f"  Macro F1:   {results['mean_macro_f1']:.4f} ± {results['std_macro_f1']:.4f}")
    print(f"  MCC:        {results['mean_mcc']:.4f} ± {results['std_mcc']:.4f}")
    print(f"  Accuracy:   {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"  Total time: {total_time:.2f} sec")
    return results


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 3,
    cv: int = 3,
    random_state: int = RANDOM_SEED,
) -> Tuple[RandomForestClassifier, Dict[str, Any], float, Dict[str, Any]]:
    """Controlled hyperparameter search for Random Forest on training data."""
    param_dist = {
        "n_estimators": [50, 100],
        "max_depth": [15, 25],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt"],
    }
    rf = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=N_JOBS)
    search = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=1,
    )
    with timer("Random Forest Hyperparameter Tuning"):
        search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, float(search.best_score_), param_dist


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 3,
    cv: int = 3,
    random_state: int = RANDOM_SEED,
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any], float, Dict[str, Any]]:
    """Controlled hyperparameter search for LightGBM on training data."""
    param_dist = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
        "colsample_bytree": [0.3, 0.5],
    }
    lgb_clf = lgb.LGBMClassifier(
        subsample=0.8,
        subsample_freq=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    search = RandomizedSearchCV(
        lgb_clf,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=1,
    )
    with timer("LightGBM Hyperparameter Tuning"):
        search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, float(search.best_score_), param_dist


def tune_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 3,
    random_state: int = RANDOM_SEED,
) -> Tuple[LinearSVC, Dict[str, Any], float, Dict[str, Any]]:
    """Controlled hyperparameter search for LinearSVC across C values on training data."""
    param_dist = {
        "C": [0.01, 0.05, 0.1],
    }
    svm_clf = LinearSVC(
        class_weight="balanced",
        dual=False,
        tol=1e-3,
        max_iter=2000,
        random_state=random_state,
    )
    search = RandomizedSearchCV(
        svm_clf,
        param_dist,
        n_iter=len(param_dist["C"]),
        cv=cv,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=1,
    )
    with timer("Linear SVM Hyperparameter Tuning"):
        search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, float(search.best_score_), param_dist

