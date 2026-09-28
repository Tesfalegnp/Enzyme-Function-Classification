"""Feature extraction and dimensionality reduction for protein sequences.

Implements:
1. Amino Acid Composition (AAC - 20D)
2. Dipeptide Composition (DPC - 400D)
3. Tripeptide Composition (TPC - 8000D sparse CountVectorizer)
4. Truncated SVD Embedding (128D, fit strictly on train)
5. Feature combination and leakage-free feature selection
"""

from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

from src.config import (
    AAC_DIM,
    DPC_DIM,
    MODELS_DIR,
    RANDOM_SEED,
    SELECT_K_FEATURES,
    STANDARD_AA,
    SVD_COMPONENTS,
)
from src.utils import save_artifact, timer

# Dipeptide vocabulary: 20 x 20 = 400
DIPEPTIDES: List[str] = [f"{a}{b}" for a, b in product(STANDARD_AA, repeat=2)]
DPC_INDEX_MAP: Dict[str, int] = {dp: idx for idx, dp in enumerate(DIPEPTIDES)}


def compute_aac(sequences: Union[List[str], pd.Series]) -> np.ndarray:
    """Compute 20-dimensional normalized Amino Acid Composition (AAC).

    Parameters
    ----------
    sequences : Iterable of protein sequence strings.

    Returns
    -------
    np.ndarray
        Shape (N, 20) float32 array with normalized frequencies.
    """
    n_seqs = len(sequences)
    aac_matrix = np.zeros((n_seqs, AAC_DIM), dtype=np.float32)

    for row_idx, seq in enumerate(sequences):
        seq_str = str(seq)
        seq_len = len(seq_str)
        if seq_len == 0:
            continue
        inv_len = 1.0 / seq_len
        for col_idx, aa in enumerate(STANDARD_AA):
            cnt = seq_str.count(aa)
            if cnt > 0:
                aac_matrix[row_idx, col_idx] = cnt * inv_len

    return aac_matrix


def compute_dpc(sequences: Union[List[str], pd.Series]) -> np.ndarray:
    """Compute 400-dimensional normalized Dipeptide Composition (DPC).

    Parameters
    ----------
    sequences : Iterable of protein sequence strings.

    Returns
    -------
    np.ndarray
        Shape (N, 400) float32 array with normalized frequencies.
    """
    n_seqs = len(sequences)
    dpc_matrix = np.zeros((n_seqs, DPC_DIM), dtype=np.float32)

    for row_idx, seq in enumerate(sequences):
        seq_str = str(seq)
        seq_len = len(seq_str)
        total_pairs = seq_len - 1
        if total_pairs <= 0:
            continue

        for i in range(total_pairs):
            pair = seq_str[i : i + 2]
            idx = DPC_INDEX_MAP.get(pair)
            if idx is not None:
                dpc_matrix[row_idx, idx] += 1.0

        dpc_matrix[row_idx] /= total_pairs

    return dpc_matrix


def _get_tripeptides(seq: Any) -> List[str]:
    """Tokenize protein sequence into overlapping 3-mers."""
    seq_str = str(seq)
    length = len(seq_str)
    if length < 3:
        return []
    return [seq_str[i : i + 3] for i in range(length - 2)]


def extract_tpc_sparse(
    train_seqs: Union[List[str], pd.Series],
    val_seqs: Optional[Union[List[str], pd.Series]] = None,
    test_seqs: Optional[Union[List[str], pd.Series]] = None,
    save_vectorizer: bool = True,
) -> Tuple[csr_matrix, Optional[csr_matrix], Optional[csr_matrix], CountVectorizer]:
    """Extract sparse Tripeptide Composition (TPC) features.

    Vectorizer is fit exclusively on the training sequences to prevent data leakage.
    """
    with timer("Tripeptide CountVectorizer Fitting"):
        vectorizer = CountVectorizer(
            analyzer=_get_tripeptides,
            dtype=np.float32,
            min_df=2,  # Prunes rare spurious non-standard k-mers
        )
        X_train_tpc = vectorizer.fit_transform(train_seqs)

    X_val_tpc = vectorizer.transform(val_seqs) if val_seqs is not None else None
    X_test_tpc = vectorizer.transform(test_seqs) if test_seqs is not None else None

    if save_vectorizer:
        save_artifact(vectorizer, MODELS_DIR / "tripeptide_vectorizer.pkl")

    return X_train_tpc, X_val_tpc, X_test_tpc, vectorizer


def reduce_tpc_svd(
    X_train_tpc: csr_matrix,
    X_val_tpc: Optional[csr_matrix] = None,
    X_test_tpc: Optional[csr_matrix] = None,
    n_components: int = SVD_COMPONENTS,
    random_state: int = RANDOM_SEED,
    save_svd: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], TruncatedSVD]:
    """Reduce high-dimensional sparse TPC to 128D dense embeddings via TruncatedSVD.

    SVD is fit exclusively on the training matrix to prevent data leakage.
    """
    with timer(f"TruncatedSVD ({n_components} components) Fitting"):
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_train_emb = svd.fit_transform(X_train_tpc).astype(np.float32)

    X_val_emb = svd.transform(X_val_tpc).astype(np.float32) if X_val_tpc is not None else None
    X_test_emb = svd.transform(X_test_tpc).astype(np.float32) if X_test_tpc is not None else None

    if save_svd:
        save_artifact(svd, MODELS_DIR / "truncated_svd.pkl")

    return X_train_emb, X_val_emb, X_test_emb, svd


def combine_feature_matrices(
    aac: np.ndarray,
    dpc: np.ndarray,
    emb: np.ndarray,
) -> np.ndarray:
    """Concatenate AAC (20D), DPC (400D), and SVD (128D) into a single 548D matrix."""
    return np.hstack([aac, dpc, emb]).astype(np.float32)


def get_feature_names() -> List[str]:
    """Generate meaningful names for all 548 features."""
    names = []
    # AAC
    names.extend([f"AAC_{aa}" for aa in STANDARD_AA])
    # DPC
    names.extend([f"DPC_{dp}" for dp in DIPEPTIDES])
    # SVD
    names.extend([f"SVD_Comp_{i+1}" for i in range(SVD_COMPONENTS)])
    return names


def select_features_train_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    k: int = SELECT_K_FEATURES,
    save_selector: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], SelectKBest]:
    """Apply SelectKBest using ANOVA F-value (f_classif).

    Selector is fit strictly on training data to prevent data leakage.
    f_classif is used instead of mutual_info_classif to maintain high discriminative
    power while avoiding the computationally prohibitive O(N^2) k-NN complexity.
    """
    actual_k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=actual_k)

    with timer(f"SelectKBest (f_classif, k={actual_k}) Fitting"):
        X_train_sel = selector.fit_transform(X_train, y_train).astype(np.float32)

    X_val_sel = selector.transform(X_val).astype(np.float32) if X_val is not None else None
    X_test_sel = selector.transform(X_test).astype(np.float32) if X_test is not None else None

    if save_selector:
        save_artifact(selector, MODELS_DIR / "select_k_best.pkl")

    return X_train_sel, X_val_sel, X_test_sel, selector
