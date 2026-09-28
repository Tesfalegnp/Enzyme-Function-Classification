"""Data cleaning, filtering, and deduplication module for SwissProt-EC."""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.config import CLEANED_DATA_PATH, DEDUP_DATA_PATH, VALID_EC_CLASSES
from src.data_validation import extract_ec_first_digit, identify_label_column, identify_sequence_column


def clean_raw_dataset(
    df_raw: pd.DataFrame,
    save_path: Path = CLEANED_DATA_PATH,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Filter raw records to valid sequence and EC class (1-6) pairs.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Combined raw DataFrame from train, dev, and test.
    save_path : Path
        Destination parquet path for cleaned sequences.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        Cleaned DataFrame with columns ['sequence', 'label'] and cleaning statistics.
    """
    initial_count = len(df_raw)
    seq_col = identify_sequence_column(df_raw)
    label_col = identify_label_column(df_raw)

    # Extract EC first digit
    ec_classes = df_raw[label_col].apply(extract_ec_first_digit)

    # Filter valid rows
    valid_mask = df_raw[seq_col].notnull() & ec_classes.notnull()
    df_cleaned = pd.DataFrame({
        "sequence": df_raw.loc[valid_mask, seq_col].astype(str),
        "label": ec_classes[valid_mask].astype(int),
    }).reset_index(drop=True)

    final_count = len(df_cleaned)
    stats = {
        "initial_rows": initial_count,
        "removed_rows": initial_count - final_count,
        "final_rows": final_count,
    }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_cleaned.to_parquet(save_path, index=False)

    return df_cleaned, stats


def deduplicate_dataset(
    df_cleaned: pd.DataFrame,
    save_path: Path = DEDUP_DATA_PATH,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove sequences with conflicting labels and exact sequence duplicates.

    Parameters
    ----------
    df_cleaned : pd.DataFrame
        Cleaned DataFrame with ['sequence', 'label'].
    save_path : Path
        Destination parquet path for deduplicated sequences.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        Deduplicated DataFrame and audit counts.
    """
    initial_count = len(df_cleaned)

    # Step 1: Remove conflicting sequences (identical sequence, different labels)
    label_counts_per_seq = df_cleaned.groupby("sequence")["label"].nunique()
    conflicting_seqs = label_counts_per_seq[label_counts_per_seq > 1].index
    num_conflicting_seqs = len(conflicting_seqs)

    if num_conflicting_seqs > 0:
        df_no_conflicts = df_cleaned[~df_cleaned["sequence"].isin(conflicting_seqs)].copy()
    else:
        df_no_conflicts = df_cleaned.copy()

    removed_conflicts = initial_count - len(df_no_conflicts)

    # Step 2: Remove exact duplicate sequences (keeping the first occurrence)
    before_exact = len(df_no_conflicts)
    df_dedup = df_no_conflicts.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
    removed_duplicates = before_exact - len(df_dedup)

    stats = {
        "initial_rows": initial_count,
        "conflicting_sequences_found": num_conflicting_seqs,
        "rows_removed_conflicts": removed_conflicts,
        "rows_removed_duplicates": removed_duplicates,
        "final_rows": len(df_dedup),
    }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_dedup.to_parquet(save_path, index=False)

    return df_dedup, stats
