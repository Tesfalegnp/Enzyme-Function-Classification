"""Unit tests for feature engineering: AAC, DPC, TPC, SVD, and selection."""

import unittest
import numpy as np
from scipy.sparse import issparse

from src.features import (
    compute_aac,
    compute_dpc,
    extract_tpc_sparse,
    reduce_tpc_svd,
    combine_feature_matrices,
    select_features_train_only,
    get_feature_names,
)


class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Sample synthetic sequences
        self.train_seqs = [
            "ACDEFGHIKLMNPQRSTVWY",
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
            "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE",
            "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK",
            "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLN",
            "EVALVALVALVALVALVALVALVALVALVALVALVALVAL",
        ]
        self.val_seqs = [
            "ACDEFGHIKLMNPQRSTVWY",
            "EVALVALVALVALVALVALVALVALVALVALVALVALVAL",
        ]

    def test_compute_aac_shape_and_sum(self):
        aac = compute_aac(self.train_seqs)
        self.assertEqual(aac.shape, (6, 20))
        self.assertEqual(aac.dtype, np.float32)
        # Frequencies must sum to ~1.0 for valid standard AA sequences
        for row in aac:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=4)

    def test_compute_dpc_shape_and_sum(self):
        dpc = compute_dpc(self.train_seqs)
        self.assertEqual(dpc.shape, (6, 400))
        self.assertEqual(dpc.dtype, np.float32)
        for row in dpc:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=4)

    def test_tpc_and_svd(self):
        X_tr_tpc, X_va_tpc, _, vec = extract_tpc_sparse(
            self.train_seqs, self.val_seqs, None, save_vectorizer=False
        )
        self.assertTrue(issparse(X_tr_tpc))
        self.assertTrue(issparse(X_va_tpc))
        self.assertEqual(X_tr_tpc.shape[0], 6)
        self.assertEqual(X_va_tpc.shape[0], 2)
        self.assertEqual(X_tr_tpc.shape[1], X_va_tpc.shape[1])

        # Test SVD reduction with small component count for test
        X_tr_emb, X_va_emb, _, svd = reduce_tpc_svd(
            X_tr_tpc, X_va_tpc, None, n_components=4, random_state=42, save_svd=False
        )
        self.assertEqual(X_tr_emb.shape, (6, 4))
        self.assertEqual(X_va_emb.shape, (2, 4))
        self.assertEqual(X_tr_emb.dtype, np.float32)

    def test_combine_and_select(self):
        aac = compute_aac(self.train_seqs)
        dpc = compute_dpc(self.train_seqs)
        emb = np.ones((6, 128), dtype=np.float32)

        combined = combine_feature_matrices(aac, dpc, emb)
        self.assertEqual(combined.shape, (6, 548))

        names = get_feature_names()
        self.assertEqual(len(names), 548)

        # Selection test
        y_train = np.array([1, 2, 3, 4, 5, 6])
        X_tr_sel, _, _, selector = select_features_train_only(
            combined, y_train, None, None, k=50, save_selector=False
        )
        self.assertEqual(X_tr_sel.shape, (6, 50))


if __name__ == "__main__":
    unittest.main()
