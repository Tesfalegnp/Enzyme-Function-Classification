"""Unit tests for model baselines, calibration, ensembles, and evaluation."""

import unittest
import numpy as np

from src.models import (
    get_random_forest_baseline,
    get_lightgbm_baseline,
    get_svm_baseline,
    get_calibrated_svm,
    run_stratified_cv,
)
from src.ensembles import (
    build_hard_voting_classifier,
    build_soft_voting_classifier,
    build_stacking_classifier,
)
from src.evaluation import compute_metrics


class TestModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic multi-class classification problem (6 classes, 30 samples, 20 features)
        self.X = np.random.randn(30, 20).astype(np.float32)
        self.y = np.array([(i % 6) + 1 for i in range(30)])

    def test_baseline_initialization(self):
        rf = get_random_forest_baseline(n_estimators=5, random_state=42)
        lgb = get_lightgbm_baseline(n_estimators=5, random_state=42)
        svm = get_svm_baseline(random_state=42)

        rf.fit(self.X, self.y)
        lgb.fit(self.X, self.y)
        svm.fit(self.X, self.y)

        self.assertEqual(len(rf.predict(self.X)), 30)
        self.assertEqual(len(lgb.predict(self.X)), 30)
        self.assertEqual(len(svm.predict(self.X)), 30)

    def test_calibrated_svm_probabilities(self):
        calibrated_svm = get_calibrated_svm(cv=2)
        calibrated_svm.fit(self.X, self.y)
        probs = calibrated_svm.predict_proba(self.X)

        self.assertEqual(probs.shape, (30, 6))
        # Each row should sum to 1.0
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(30), rtol=1e-5)

    def test_ensembles(self):
        rf = get_random_forest_baseline(n_estimators=5, random_state=42)
        lgb = get_lightgbm_baseline(n_estimators=5, random_state=42)
        svm = get_svm_baseline(random_state=42)

        hard_vote = build_hard_voting_classifier(rf, lgb, svm)
        hard_vote.fit(self.X, self.y)
        preds_hard = hard_vote.predict(self.X)
        self.assertEqual(len(preds_hard), 30)

        soft_vote = build_soft_voting_classifier(rf, lgb, svm)
        soft_vote.fit(self.X, self.y)
        preds_soft = soft_vote.predict(self.X)
        self.assertEqual(len(preds_soft), 30)

        stacking = build_stacking_classifier(rf, lgb, svm, cv=2, random_state=42)
        stacking.fit(self.X, self.y)
        preds_stack = stacking.predict(self.X)
        self.assertEqual(len(preds_stack), 30)

    def test_compute_metrics(self):
        y_true = np.array([1, 2, 3, 4, 5, 6])
        y_pred = np.array([1, 2, 3, 4, 5, 1])  # 5/6 correct
        metrics = compute_metrics(y_true, y_pred)

        self.assertIn("Macro F1", metrics)
        self.assertIn("MCC", metrics)
        self.assertIn("Accuracy", metrics)
        self.assertAlmostEqual(metrics["Accuracy"], 5.0 / 6.0, places=4)


if __name__ == "__main__":
    unittest.main()
