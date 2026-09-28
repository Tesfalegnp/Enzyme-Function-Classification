"""Unit tests for data loading, validation, cleaning, and splitting."""

import unittest
import pandas as pd
import numpy as np

from src.data_validation import (
    extract_ec_first_digit,
    identify_sequence_column,
    identify_label_column,
    analyze_cross_split_leakage,
)
from src.preprocessing import clean_raw_dataset, deduplicate_dataset
from src.splitting import split_dataset


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Synthetic mini dataset with edge cases:
        # - valid sequences
        # - non-standard amino acids
        # - exact duplicate sequence with same label
        # - conflicting sequence with different labels
        # - invalid EC label
        self.raw_df = pd.DataFrame({
            "seq": [
                "ACDEFGHIKLMNPQRSTVWY",
                "ACDEFGHIKLMNPQRSTVWY",  # exact duplicate
                "MKWVTFISLLLLFSSAYSRG",
                "MKWVTFISLLLLFSSAYSRG",  # conflicting label below
                "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
                "AAAAAAA",  # invalid EC
            ],
            "ec_number": [
                "EC:1.1.1.1",
                "EC:1.1.1.1",
                "EC:2.7.1.1",
                "EC:3.1.2.1",  # conflict with row 2
                "4.1.2.3",
                "EC:9.9.9.9",  # invalid EC (outside 1-6)
            ],
        })

    def test_identify_columns(self):
        self.assertEqual(identify_sequence_column(self.raw_df), "seq")
        self.assertEqual(identify_label_column(self.raw_df), "ec_number")

    def test_extract_ec_first_digit(self):
        self.assertEqual(extract_ec_first_digit("EC:1.1.1.1"), 1)
        self.assertEqual(extract_ec_first_digit("2.7.1.1"), 2)
        self.assertEqual(extract_ec_first_digit(["EC:3.4.21.4"]), 3)
        self.assertEqual(extract_ec_first_digit("6.3.2.1"), 6)
        self.assertIsNone(extract_ec_first_digit("EC:7.1.1.1"))
        self.assertIsNone(extract_ec_first_digit(None))
        self.assertIsNone(extract_ec_first_digit([]))

    def test_clean_raw_dataset(self):
        cleaned_df, stats = clean_raw_dataset(self.raw_df, save_path=None)
        # Should drop row 5 with EC:9.9.9.9
        self.assertEqual(len(cleaned_df), 5)
        self.assertIn("sequence", cleaned_df.columns)
        self.assertIn("label", cleaned_df.columns)
        self.assertTrue(all(cleaned_df["label"].between(1, 6)))

    def test_deduplicate_dataset(self):
        cleaned_df, _ = clean_raw_dataset(self.raw_df, save_path=None)
        dedup_df, stats = deduplicate_dataset(cleaned_df, save_path=None)

        # Conflicting sequence "MKWVTFISLLLLFSSAYSRG" (2 rows) must be removed
        # Duplicate sequence "ACDEFGHIKLMNPQRSTVWY" (2 rows) reduced to 1 row
        # Row 4 remains
        # Total remaining = 1 + 1 = 2 rows
        self.assertEqual(stats["conflicting_sequences_found"], 1)
        self.assertEqual(stats["rows_removed_conflicts"], 2)
        self.assertEqual(stats["rows_removed_duplicates"], 1)
        self.assertEqual(len(dedup_df), 2)
        self.assertEqual(dedup_df["sequence"].nunique(), len(dedup_df))

    def test_split_dataset_no_leakage(self):
        # Create dataset with all 6 classes and 60 samples
        seqs = [f"SEQ_{i}_" + "ACDEFGHIKLMNPQRSTVWY" for i in range(60)]
        labels = [(i % 6) + 1 for i in range(60)]
        df = pd.DataFrame({"sequence": seqs, "label": labels})

        df_train, df_val, df_test, summary = split_dataset(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42, save=False
        )

        train_s = set(df_train["sequence"])
        val_s = set(df_val["sequence"])
        test_s = set(df_test["sequence"])

        # No sequence overlap
        self.assertEqual(len(train_s.intersection(val_s)), 0)
        self.assertEqual(len(train_s.intersection(test_s)), 0)
        self.assertEqual(len(val_s.intersection(test_s)), 0)

        # All classes present in every split
        self.assertEqual(set(df_train["label"].unique()), set(range(1, 7)))
        self.assertEqual(set(df_val["label"].unique()), set(range(1, 7)))
        self.assertEqual(set(df_test["label"].unique()), set(range(1, 7)))


if __name__ == "__main__":
    unittest.main()
