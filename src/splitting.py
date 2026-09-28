"""Stratified dataset partitioning module for reproducible 70/15/15 splits."""

from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_SEED,
    TEST_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
    VAL_SPLIT_PATH,
    VALID_EC_CLASSES,
)


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = RANDOM_SEED,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Partition deduplicated dataset into stratified train, val, and test splits.

    Ensures zero cross-split leakage, deterministic reproduction, and preserved
    class distributions.
    """
    total = len(df)
    temp_ratio = val_ratio + test_ratio
    val_within_temp = val_ratio / temp_ratio

    X = df["sequence"]
    y = df["label"]

    # First split: 70% Train, 30% Temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_ratio,
        stratify=y,
        random_state=random_state,
    )

    # Second split: 15% Val, 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_within_temp),
        stratify=y_temp,
        random_state=random_state,
    )

    df_train = pd.DataFrame({"sequence": X_train, "label": y_train}).reset_index(drop=True)
    df_val = pd.DataFrame({"sequence": X_val, "label": y_val}).reset_index(drop=True)
    df_test = pd.DataFrame({"sequence": X_test, "label": y_test}).reset_index(drop=True)

    # Verify class representation
    for split_name, df_split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        classes_present = set(df_split["label"].unique())
        expected_classes = set(VALID_EC_CLASSES)
        if classes_present != expected_classes:
            raise ValueError(
                f"{split_name} split missing classes: {expected_classes - classes_present}"
            )

    summary = {
        "total_rows": total,
        "train_rows": len(df_train),
        "train_pct": (len(df_train) / total) * 100,
        "val_rows": len(df_val),
        "val_pct": (len(df_val) / total) * 100,
        "test_rows": len(df_test),
        "test_pct": (len(df_test) / total) * 100,
        "classes_verified": list(VALID_EC_CLASSES),
    }

    if save:
        TRAIN_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_train.to_parquet(TRAIN_SPLIT_PATH, index=False)
        df_val.to_parquet(VAL_SPLIT_PATH, index=False)
        df_test.to_parquet(TEST_SPLIT_PATH, index=False)

    return df_train, df_val, df_test, summary
