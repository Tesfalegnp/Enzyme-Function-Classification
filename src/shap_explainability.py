"""Optional SHAP (SHapley Additive exPlanations) Explainability Module.

Provides game-theoretic model interpretability for tree-based classifiers
(Random Forest and LightGBM) in the Enzyme Function Classification pipeline.

Core Capabilities:
1. Global Feature Importance:
   - Mean absolute SHAP value rankings across representative sample populations.
   - Beeswarm/summary plots detailing directional feature effects on decision boundaries.
   - Bar-style SHAP importance plots and CSV export.
2. Local Prediction Explanations:
   - Sample-level waterfall/contribution decomposition for single sequence predictions.
   - Clear identification of features pushing toward (supporting) or away from (opposing)
     the predicted top-level EC class (EC 1–6).
3. Representative Sampling:
   - Configurable sample sizes (default: 200) to ensure memory and runtime safety.
4. Robust Feature Recovery:
   - Preserves biological identifiers: AAC (amino acid composition), DPC (dipeptide),
     and SVD-reduced tripeptide components.

Scientific Interpretation Note:
    SHAP values quantify mathematical contributions of engineered numerical features
    toward the classifier's internal log-odds or probability partition.
    They do NOT constitute biological causation or experimentally verified catalytic mechanisms.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    VALID_EC_CLASSES,
)
from src.features import get_feature_names

# Optional SHAP import guard
try:
    import shap
    _SHAP_AVAILABLE = True
    _SHAP_VERSION = shap.__version__
except ImportError:
    shap = None
    _SHAP_AVAILABLE = False
    _SHAP_VERSION = None


def is_shap_available() -> bool:
    """Check if the optional SHAP library is installed in the current environment."""
    return _SHAP_AVAILABLE


def get_shap_version() -> Optional[str]:
    """Return the installed SHAP version, or None if unavailable."""
    return _SHAP_VERSION


def get_selected_feature_names(selector: Optional[Any] = None) -> List[str]:
    """Recover the exact 500 feature names corresponding to the SelectKBest transformation.

    Parameters
    ----------
    selector : Optional[SelectKBest]
        Fitted SelectKBest transformer. If None, loaded from models/saved/select_k_best.pkl.

    Returns
    -------
    List[str]
        List of 500 feature names in exact column order.
    """
    all_names = get_feature_names()
    if selector is None:
        import joblib
        sel_path = MODELS_DIR / "select_k_best.pkl"
        if not sel_path.exists():
            return [f"Feature_{i}" for i in range(500)]
        selector = joblib.load(sel_path)

    support = selector.get_support()
    selected_names = [all_names[i] for i, s in enumerate(support) if s]
    return selected_names


def get_tree_explainer(model: Any) -> Any:
    """Create a SHAP TreeExplainer for a fitted tree-based model (Random Forest or LightGBM).

    Parameters
    ----------
    model : BaseEstimator
        Fitted RandomForestClassifier or LGBMClassifier.

    Returns
    -------
    shap.TreeExplainer
        TreeExplainer instance.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Please install it via `pip install 'shap>=0.44.0'` "
            "to enable tree model explainability."
        )
    return shap.TreeExplainer(model)


def extract_multiclass_shap_values(
    raw_shap_output: Any,
    class_idx: int,
) -> np.ndarray:
    """Robustly extract SHAP values for a specific class index from multiclass outputs.

    Handles different SHAP API formats:
    - shap._explanation.Explanation (3D values: (N, n_features, n_classes))
    - list of np.ndarray of length n_classes, each of shape (N, n_features)
    - 3D np.ndarray of shape (N, n_features, n_classes) or (n_classes, N, n_features)

    Parameters
    ----------
    raw_shap_output : Any
        Output from TreeExplainer(model)(X) or explainer.shap_values(X).
    class_idx : int
        Zero-based index of the target class (0 = EC 1, ..., 5 = EC 6).

    Returns
    -------
    np.ndarray
        Array of shape (N, n_features) containing SHAP values for target class.
    """
    # 1. If Explanation object
    if hasattr(raw_shap_output, "values"):
        vals = raw_shap_output.values
        if vals.ndim == 3:
            # Standard shape: (N, n_features, n_classes)
            if vals.shape[2] > class_idx:
                return vals[:, :, class_idx]
            elif vals.shape[0] > class_idx:
                return vals[class_idx, :, :]
        elif vals.ndim == 2:
            return vals
        raise ValueError(f"Unexpected Explanation values dimension: {vals.shape}")

    # 2. If list of ndarrays
    if isinstance(raw_shap_output, (list, tuple)):
        if class_idx < len(raw_shap_output):
            return np.asarray(raw_shap_output[class_idx])
        raise IndexError(f"Class index {class_idx} out of bounds for list of length {len(raw_shap_output)}")

    # 3. If raw 3D ndarray
    arr = np.asarray(raw_shap_output)
    if arr.ndim == 3:
        if arr.shape[2] == len(VALID_EC_CLASSES):
            return arr[:, :, class_idx]
        elif arr.shape[0] == len(VALID_EC_CLASSES):
            return arr[class_idx, :, :]
    elif arr.ndim == 2:
        return arr

    raise ValueError(f"Unable to extract class {class_idx} from SHAP output of shape {arr.shape}")


def load_representative_sample(
    sample_size: int = 200,
    random_state: int = RANDOM_SEED,
    data_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a reproducible, representative stratified subsample of test features.

    Avoids evaluating SHAP on all 31,841 samples, ensuring memory and runtime safety.

    Parameters
    ----------
    sample_size : int, default 200
        Number of samples to draw.
    random_state : int, default 42
        Random seed for deterministic sampling.
    data_path : Optional[Path]
        Path to processed feature array (.npy). If None, defaults to X_test_final.npy.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_sub, y_sub) where X_sub is (sample_size, 500) and y_sub is (sample_size,).
    """
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "processed" / "X_test_final.npy"

    label_path = PROJECT_ROOT / "data" / "processed" / "test_split.parquet"

    if not data_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Processed test data missing. Expected {data_path} and {label_path}."
        )

    X_full = np.load(data_path)
    df_labels = pd.read_parquet(label_path)
    y_full = df_labels["label"].values

    n_samples = len(X_full)
    actual_sample_size = min(sample_size, n_samples)

    rng = np.random.RandomState(random_state)

    # Stratified subsampling
    indices = []
    classes = np.unique(y_full)
    per_class = max(1, actual_sample_size // len(classes))

    for cls in classes:
        cls_idx = np.where(y_full == cls)[0]
        chosen = rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False)
        indices.extend(chosen)

    # Pad or trim to exact sample_size if needed
    if len(indices) < actual_sample_size:
        remaining_pool = list(set(range(n_samples)) - set(indices))
        needed = actual_sample_size - len(indices)
        extra = rng.choice(remaining_pool, size=needed, replace=False)
        indices.extend(extra)
    elif len(indices) > actual_sample_size:
        indices = rng.choice(indices, size=actual_sample_size, replace=False)

    indices = np.array(sorted(indices))
    return X_full[indices].astype(np.float32), y_full[indices]


def compute_global_shap_analysis(
    model: Any,
    X_sample: np.ndarray,
    feature_names: Optional[List[str]] = None,
    model_name: str = "Random Forest",
    top_n: int = 20,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute global SHAP explainability on a representative sample.

    Calculates:
    1. Global mean absolute SHAP values across all classes and features.
    2. Summary beeswarm plot (for top features of dominant class or multi-class).
    3. Global horizontal bar plot of top N features.
    4. Tabular CSV export.

    Parameters
    ----------
    model : BaseEstimator
        Trained Random Forest or LightGBM model.
    X_sample : np.ndarray
        Representative feature matrix (e.g. shape (200, 500)).
    feature_names : Optional[List[str]]
        Names for the 500 features. Recovered automatically if None.
    model_name : str
        Human-readable name ("Random Forest" or "LightGBM").
    top_n : int
        Number of top features to report and plot.
    output_dir : Optional[Path]
        Directory to save figures and metrics. Defaults to results/figures and results/metrics.

    Returns
    -------
    Dict[str, Any]
        Dictionary with importance DataFrame, paths to generated figures and metrics.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError("SHAP is not available. Install via `pip install 'shap>=0.44.0'`.")

    if feature_names is None:
        feature_names = get_selected_feature_names()

    model_slug = "rf" if "forest" in model_name.lower() else "lgb"
    fig_dir = output_dir if output_dir is not None else FIGURES_DIR
    met_dir = output_dir if output_dir is not None else METRICS_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    explainer = get_tree_explainer(model)
    raw_exp = explainer(X_sample)

    # Compute global importance: mean absolute SHAP value across samples and classes
    # If raw_exp has values of shape (N, n_features, n_classes)
    if hasattr(raw_exp, "values"):
        shap_vals = raw_exp.values
        if shap_vals.ndim == 3:
            # Mean |SHAP| across all classes and samples
            # Shape: (n_features,)
            global_mean_abs = np.mean(np.abs(shap_vals), axis=(0, 2))
            # Also per-class mean |SHAP|: shape (n_classes, n_features)
            per_class_abs = np.mean(np.abs(shap_vals), axis=0)  # (n_features, n_classes)
        else:
            global_mean_abs = np.mean(np.abs(shap_vals), axis=0)
            per_class_abs = None
    else:
        # Fallback list of arrays
        val_list = [extract_multiclass_shap_values(raw_exp, c) for c in range(len(VALID_EC_CLASSES))]
        global_mean_abs = np.mean([np.mean(np.abs(v), axis=0) for v in val_list], axis=0)
        per_class_abs = None

    # Rank features by global mean |SHAP|
    ranked_indices = np.argsort(global_mean_abs)[::-1]
    top_indices = ranked_indices[:top_n]

    ranking_records = []
    for rank, idx in enumerate(top_indices, start=1):
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        ranking_records.append({
            "Rank": rank,
            "Feature": feat_name,
            "Mean_Abs_SHAP": float(global_mean_abs[idx]),
            "Feature_Type": (
                "AAC (Amino Acid Comp)" if feat_name.startswith("AAC_")
                else "DPC (Dipeptide Comp)" if feat_name.startswith("DPC_")
                else "SVD (Tripeptide Motif)" if feat_name.startswith("SVD_")
                else "Engineered Feature"
            ),
        })

    df_importance = pd.DataFrame(ranking_records)

    # 1. Save CSV metric table
    csv_path = met_dir / f"shap_{model_slug}_feature_importance.csv"
    df_importance.to_csv(csv_path, index=False)

    # 2. Bar plot of Global SHAP Importance
    bar_fig_path = fig_dir / f"shap_{model_slug}_bar.png"
    plt.figure(figsize=(10, 6))
    bar_colors = ["#0284c7" if "rf" in model_slug else "#059669"] * top_n
    plt.barh(range(top_n - 1, -1, -1), df_importance["Mean_Abs_SHAP"].values, color=bar_colors, align="center")
    plt.yticks(range(top_n - 1, -1, -1), df_importance["Feature"].values, fontsize=9.5)
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Decision)", fontsize=10)
    plt.title(f"Global SHAP Feature Importance (Top {top_n}) — {model_name}", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(bar_fig_path, dpi=300)
    plt.close()

    # 3. Beeswarm/Summary Plot
    summary_fig_path = fig_dir / f"shap_{model_slug}_summary.png"
    plt.figure(figsize=(11, 7))

    # For summary plot, display values for the top class or flattened/first dominant class
    if hasattr(raw_exp, "values") and raw_exp.values.ndim == 3:
        # Create an Explanation object for the top contributing class or class 0 (Oxidoreductases)
        exp_slice = raw_exp[:, :, 0]
        exp_slice.feature_names = feature_names
        shap.plots.beeswarm(exp_slice, max_display=top_n, show=False)
    else:
        # Fallback to standard summary plot
        v0 = extract_multiclass_shap_values(raw_exp, 0)
        shap.summary_plot(v0, X_sample, feature_names=feature_names, max_display=top_n, show=False)

    plt.title(f"SHAP Summary (Beeswarm) Plot — {model_name} (Representative Sample N={len(X_sample)})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(summary_fig_path, dpi=300)
    plt.close()

    return {
        "model_name": model_name,
        "sample_size": len(X_sample),
        "importance_df": df_importance,
        "bar_plot_path": bar_fig_path,
        "summary_plot_path": summary_fig_path,
        "csv_metrics_path": csv_path,
    }


def explain_prediction_local(
    model: Any,
    X_single: np.ndarray,
    predicted_ec: int,
    feature_names: Optional[List[str]] = None,
    top_n: int = 10,
    model_name: str = "Classifier",
) -> Dict[str, Any]:
    """Compute local SHAP explanation for an individual enzyme sequence prediction.

    Parameters
    ----------
    model : BaseEstimator
        Trained model (Random Forest or LightGBM).
    X_single : np.ndarray
        Feature matrix for one sample, shape (1, 500) or (500,).
    predicted_ec : int
        Predicted top-level EC class (1 to 6).
    feature_names : Optional[List[str]]
        Names for the 500 features.
    top_n : int, default 10
        Number of top supporting and opposing features to report.
    model_name : str
        Name of the classifier architecture.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing supporting features, opposing features, base value,
        predicted class label, and raw SHAP values.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError("SHAP is not available. Install via `pip install 'shap>=0.44.0'`.")

    if feature_names is None:
        feature_names = get_selected_feature_names()

    if X_single.ndim == 1:
        X_mat = X_single.reshape(1, -1)
    else:
        X_mat = X_single

    # Map EC (1..6) to zero-based class index (0..5)
    class_idx = max(0, min(predicted_ec - 1, len(VALID_EC_CLASSES) - 1))

    explainer = get_tree_explainer(model)
    raw_exp = explainer(X_mat)

    # Extract SHAP values for target class
    class_shap_vals = extract_multiclass_shap_values(raw_exp, class_idx)[0]  # shape (500,)
    sample_feat_vals = X_mat[0]  # shape (500,)

    # Base value (expected value for target class)
    base_val = 0.0
    if hasattr(raw_exp, "base_values"):
        bv = raw_exp.base_values
        if bv.ndim == 2 and bv.shape[1] > class_idx:
            base_val = float(bv[0, class_idx])
        elif bv.ndim == 1:
            base_val = float(bv[0])
    elif hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and len(ev) > class_idx:
            base_val = float(ev[class_idx])
        else:
            base_val = float(ev)

    # Separate supporting (positive SHAP pushing towards target class)
    # and opposing (negative SHAP pushing away from target class)
    pos_indices = np.where(class_shap_vals > 0)[0]
    neg_indices = np.where(class_shap_vals < 0)[0]

    # Sort descending by magnitude
    top_pos_idx = pos_indices[np.argsort(class_shap_vals[pos_indices])[::-1]][:top_n]
    top_neg_idx = neg_indices[np.argsort(np.abs(class_shap_vals[neg_indices]))[::-1]][:top_n]

    supporting_features = []
    for idx in top_pos_idx:
        fname = feature_names[idx] if idx < len(feature_names) else f"Feat_{idx}"
        supporting_features.append({
            "feature": fname,
            "feature_value": float(sample_feat_vals[idx]),
            "shap_value": float(class_shap_vals[idx]),
            "description": _describe_feature(fname, float(sample_feat_vals[idx]), float(class_shap_vals[idx]), predicted_ec),
        })

    opposing_features = []
    for idx in top_neg_idx:
        fname = feature_names[idx] if idx < len(feature_names) else f"Feat_{idx}"
        opposing_features.append({
            "feature": fname,
            "feature_value": float(sample_feat_vals[idx]),
            "shap_value": float(class_shap_vals[idx]),
            "description": _describe_feature(fname, float(sample_feat_vals[idx]), float(class_shap_vals[idx]), predicted_ec),
        })

    return {
        "model_name": model_name,
        "predicted_ec": predicted_ec,
        "class_index": class_idx,
        "base_value": base_val,
        "supporting_features": supporting_features,
        "opposing_features": opposing_features,
        "raw_shap_values": class_shap_vals,
        "feature_values": sample_feat_vals,
        "feature_names": feature_names,
    }


def _describe_feature(feature_name: str, feat_val: float, shap_val: float, target_ec: int) -> str:
    """Provide a scientifically objective description of feature contribution."""
    impact_dir = "contributed positively toward" if shap_val > 0 else "reduced evidence for"

    if feature_name.startswith("AAC_"):
        aa = feature_name.split("_")[1]
        return f"Amino acid '{aa}' composition ({feat_val*100:.2f}%) {impact_dir} EC {target_ec} prediction."
    elif feature_name.startswith("DPC_"):
        dp = feature_name.split("_")[1]
        return f"Dipeptide '{dp}' frequency ({feat_val*100:.3f}%) {impact_dir} EC {target_ec} prediction."
    elif feature_name.startswith("SVD_"):
        comp = feature_name.split("_")[-1]
        return f"SVD reduced motif component {comp} (value {feat_val:.4f}) {impact_dir} EC {target_ec} prediction."
    return f"Feature {feature_name} (value {feat_val:.4f}) {impact_dir} EC {target_ec} prediction."


def plot_local_waterfall(
    local_exp: Dict[str, Any],
    top_n: int = 10,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate a clean, publication-grade local SHAP contribution plot.

    Parameters
    ----------
    local_exp : Dict[str, Any]
        Output dictionary from explain_prediction_local.
    top_n : int, default 10
        Number of most influential features to display.
    save_path : Optional[Path]
        Optional path to save PNG figure.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    predicted_ec = local_exp["predicted_ec"]
    model_name = local_exp["model_name"]
    shap_vals = local_exp["raw_shap_values"]
    feat_vals = local_exp["feature_values"]
    feat_names = local_exp["feature_names"]

    # Select top features by absolute SHAP value
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    plot_shap = shap_vals[top_idx]
    plot_names = [f"{feat_names[i]} ({feat_vals[i]:.3f})" for i in top_idx]

    # Invert order for horizontal top-down presentation
    plot_shap = plot_shap[::-1]
    plot_names = plot_names[::-1]

    colors = ["#059669" if val > 0 else "#dc2626" for val in plot_shap]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(range(len(plot_shap)), plot_shap, color=colors, align="center", alpha=0.85)
    ax.set_yticks(range(len(plot_shap)))
    ax.set_yticklabels(plot_names, fontsize=9.5)
    ax.axvline(0, color="#64748b", linewidth=0.8, linestyle="--")

    ax.set_xlabel(f"SHAP Contribution Value (Impact on EC {predicted_ec})", fontsize=10)
    ax.set_title(
        f"Local SHAP Feature Contributions — Predicted EC {predicted_ec} ({model_name})\n"
        f"[Green: Supports EC {predicted_ec} | Red: Opposes EC {predicted_ec}]",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig
