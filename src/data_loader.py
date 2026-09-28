"""Data loading functions for SwissProt-EC raw and processed parquet files."""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    TRAIN_PARQUET,
    DEV_PARQUET,
    TEST_PARQUET,
    TRAIN_SPLIT_PATH,
    VAL_SPLIT_PATH,
    TEST_SPLIT_PATH,
    CLEANED_DATA_PATH,
    DEDUP_DATA_PATH,
)


def get_raw_file_path(filename: str) -> Path:
    """Resolve raw parquet file path handling potential nested directory layouts."""
    path = RAW_DATA_DIR / filename
    if not path.exists():
        nested = RAW_DATA_DIR / "swissprot-ec" / filename
        if nested.exists():
            return nested
    return path


def verify_raw_files() -> Dict[str, Tuple[bool, float]]:
    """Check existence and size of required raw dataset parquet files."""
    results = {}
    for fname in [TRAIN_PARQUET, DEV_PARQUET, TEST_PARQUET]:
        fpath = get_raw_file_path(fname)
        exists = fpath.exists()
        size_mb = (fpath.stat().st_size / (1024 * 1024)) if exists else 0.0
        results[fname] = (exists, size_mb)
    return results


def load_raw_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load original SwissProt-EC train, dev, and test parquet files."""
    train_path = get_raw_file_path(TRAIN_PARQUET)
    dev_path = get_raw_file_path(DEV_PARQUET)
    test_path = get_raw_file_path(TEST_PARQUET)

    for p, name in [(train_path, "Train"), (dev_path, "Dev"), (test_path, "Test")]:
        if not p.exists():
            raise FileNotFoundError(f"Raw {name} dataset not found at: {p}")

    df_train = pd.read_parquet(train_path)
    df_dev = pd.read_parquet(dev_path)
    df_test = pd.read_parquet(test_path)
    return df_train, df_dev, df_test


def load_combined_raw() -> pd.DataFrame:
    """Load and concatenate all raw splits into a single master DataFrame."""
    df_train, df_dev, df_test = load_raw_datasets()
    df_all = pd.concat([df_train, df_dev, df_test], ignore_index=True)
    return df_all


def load_processed_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the final stratified 70/15/15 train, validation, and test splits."""
    if not (TRAIN_SPLIT_PATH.exists() and VAL_SPLIT_PATH.exists() and TEST_SPLIT_PATH.exists()):
        raise FileNotFoundError(
            "Processed splits not found. Run preprocessing and splitting stages first."
        )
    df_train = pd.read_parquet(TRAIN_SPLIT_PATH)
    df_val = pd.read_parquet(VAL_SPLIT_PATH)
    df_test = pd.read_parquet(TEST_SPLIT_PATH)
    return df_train, df_val, df_test
