"""Publication-quality visualization module for EDA, metrics, curves, and importances."""

from pathlib import Path
from typing import Any, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from src.config import FIGURES_DIR, VALID_EC_CLASSES

# Set clean aesthetic style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12


def plot_eda_distributions(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot EC class distributions and sequence length distributions across splits."""
    if save_path is None:
        save_path = FIGURES_DIR / "eda_class_and_length_distribution.png"

    df_tr = df_train.copy()
    df_tr["Split"] = "Train"
    df_va = df_val.copy()
    df_va["Split"] = "Validation"
    df_te = df_test.copy()
    df_te["Split"] = "Test"
    combined = pd.concat([df_tr, df_va, df_te], ignore_index=True)
    combined["length"] = combined["sequence"].astype(str).str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Class Distribution
    sns.countplot(data=combined, x="label", hue="Split", ax=axes[0], palette="Set2")
    axes[0].set_title("EC Primary Class Distribution Across Splits")
    axes[0].set_xlabel("EC Class (1–6)")
    axes[0].set_ylabel("Sequence Count")

    # 2. Sequence Length by Class
    sns.boxplot(data=combined, x="label", y="length", hue="Split", ax=axes[1], palette="Set2")
    axes[1].set_title("Sequence Length Distribution by EC Class")
    axes[1].set_xlabel("EC Class (1–6)")
    axes[1].set_ylabel("Sequence Length (aa, log scale)")
    axes[1].set_yscale("log")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: np.ndarray,
    model_name: str = "Stacking",
    save_path: Optional[Path] = None,
) -> Path:
    """Plot normalized-by-true-class confusion matrix."""
    if save_path is None:
        save_path = FIGURES_DIR / f"normalized_confusion_matrix_{model_name.lower().replace(' ', '_')}.png"

    classes = sorted(list(VALID_EC_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.title(f"Normalized Confusion Matrix — {model_name} (Test Set)")
    plt.xlabel("Predicted EC Class")
    plt.ylabel("True EC Class")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_multiclass_roc(
    y_true: Union[np.ndarray, pd.Series],
    y_score: np.ndarray,
    model_name: str = "Ensemble",
    save_path: Optional[Path] = None,
) -> Path:
    """Plot One-vs-Rest ROC curves and compute AUC for each enzyme class."""
    if save_path is None:
        save_path = FIGURES_DIR / "ovr_roc_curves.png"

    classes = sorted(list(VALID_EC_CLASSES))
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Class {classes[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"One-vs-Rest ROC Curves — {model_name} (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_precision_recall_curves(
    y_true: Union[np.ndarray, pd.Series],
    y_score: np.ndarray,
    model_name: str = "Ensemble",
    save_path: Optional[Path] = None,
) -> Path:
    """Plot multi-class Precision-Recall curves and compute Average Precision."""
    if save_path is None:
        save_path = FIGURES_DIR / "precision_recall_curves.png"

    from sklearn.metrics import average_precision_score

    classes = sorted(list(VALID_EC_CLASSES))
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, lw=2, label=f"Class {classes[i]} (AP = {ap:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves by EC Class — {model_name} (Test Set)")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_feature_importances(
    importances: np.ndarray,
    feature_names: List[str],
    model_name: str = "LightGBM",
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot top N feature importances with descriptive names."""
    if save_path is None:
        save_path = FIGURES_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"

    indices = np.argsort(importances)[::-1][:top_n]
    top_scores = importances[indices]
    top_labels = [feature_names[i] if i < len(feature_names) else f"Feat_{i}" for i in indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n - 1, -1, -1), top_scores, color="teal", align="center")
    plt.yticks(range(top_n - 1, -1, -1), top_labels)
    plt.xlabel("Relative Importance Score")
    plt.title(f"Top {top_n} Feature Importances ({model_name})")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
