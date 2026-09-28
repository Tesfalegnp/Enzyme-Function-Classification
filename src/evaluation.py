"""Evaluation metrics, multi-model benchmarks, and error analysis."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

from src.config import METRICS_DIR, VALID_EC_CLASSES
from src.utils import timer


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute required multi-class evaluation metrics.

    Metrics: Macro F1, MCC, Accuracy, and Macro AUPRC.
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))

    auprc = np.nan
    if y_prob is not None:
        try:
            y_bin = label_binarize(y_true, classes=VALID_EC_CLASSES)
            if y_bin.shape[1] == y_prob.shape[1]:
                auprc = float(average_precision_score(y_bin, y_prob, average="macro"))
        except Exception:
            auprc = np.nan

    p_macro, r_macro, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    return {
        "Macro F1": macro_f1,
        "MCC": mcc,
        "AUPRC": auprc,
        "Accuracy": acc,
        "Macro Precision": float(p_macro),
        "Macro Recall": float(r_macro),
    }


def evaluate_models_on_test(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: Union[np.ndarray, pd.Series],
    save_parquet: bool = True,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Evaluate a dictionary of models on the untouched held-out test set."""
    y_test_arr = np.array(y_test)
    records = []

    print("\n--- Evaluating Models on Held-Out Test Set ---")
    for name, model in models.items():
        with timer(f"Evaluating {name}"):
            preds = model.predict(X_test)
            probs = None
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X_test)
                except Exception:
                    probs = None

            m = compute_metrics(y_test_arr, preds, probs)
            records.append({
                "Model": name,
                "Macro F1": m["Macro F1"],
                "MCC": m["MCC"],
                "AUPRC": m["AUPRC"],
                "Accuracy": m["Accuracy"],
                "Macro Precision": m["Macro Precision"],
                "Macro Recall": m["Macro Recall"],
            })
            print(
                f"[{name}] Macro F1: {m['Macro F1']:.4f} | MCC: {m['MCC']:.4f} | AUPRC: {m['AUPRC']:.4f} | Acc: {m['Accuracy']:.4f}"
            )

    df_results = pd.DataFrame(records)

    if save_parquet:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        df_results.to_parquet(METRICS_DIR / "final_test_metrics.parquet", index=False)
    if save_csv:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(METRICS_DIR / "final_test_metrics.csv", index=False)

    return df_results


def conduct_error_analysis(
    sequences: Union[List[str], pd.Series],
    y_true: Union[np.ndarray, pd.Series],
    y_pred: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Perform detailed error analysis on sequence length vs misclassification and confusion pairs."""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    seqs = [str(s) for s in sequences]

    df_err = pd.DataFrame({
        "sequence": seqs,
        "true_label": y_t,
        "pred_label": y_p,
    })
    df_err["is_correct"] = df_err["true_label"] == df_err["pred_label"]
    df_err["length"] = df_err["sequence"].str.len()

    # Confusion pairs
    misclassified = df_err[~df_err["is_correct"]]
    confusion_pairs = (
        misclassified.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    stats = {
        "total_samples": len(df_err),
        "correct_predictions": int(df_err["is_correct"].sum()),
        "incorrect_predictions": int((~df_err["is_correct"]).sum()),
        "accuracy": float(df_err["is_correct"].mean()),
        "mean_len_correct": float(df_err[df_err["is_correct"]]["length"].mean()),
        "mean_len_incorrect": float(df_err[~df_err["is_correct"]]["length"].mean()) if len(misclassified) > 0 else 0.0,
    }

    return df_err, confusion_pairs, stats
