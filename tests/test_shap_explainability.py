"""Unit tests for the optional SHAP explainability module.

Validates:
1. SHAP module import and availability check.
2. Creation of TreeExplainer instances for Random Forest and LightGBM.
3. Dimension verification of recovered feature names (500 features).
4. Fast local prediction explanations on single samples.
5. Multiclass handling across EC 1–6 classes.
6. Verification that SHAP computations do not mutate or alter model parameters.
7. Seamless interoperability with existing inference without breaking the core pipeline.
"""

import copy
import unittest
import numpy as np

from src.inference import load_inference_artifacts, predict_sequence
from src.shap_explainability import (
    is_shap_available,
    get_shap_version,
    get_selected_feature_names,
    get_tree_explainer,
    explain_prediction_local,
    extract_multiclass_shap_values,
    plot_local_waterfall,
)


class TestShapExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shap_installed = is_shap_available()
        cls.artifacts = load_inference_artifacts()
        cls.feature_names = get_selected_feature_names(cls.artifacts["selector"])
        # Fast synthetic feature vector of 500 features
        rng = np.random.RandomState(42)
        cls.mock_sample = rng.randn(1, 500).astype(np.float32)

    def test_shap_availability(self):
        """Test SHAP installation and version detection."""
        self.assertTrue(self.shap_installed, "SHAP should be installed in the environment.")
        version = get_shap_version()
        self.assertIsNotNone(version)

    def test_feature_names_dimension(self):
        """Test that recovered feature names match exactly 500 features."""
        self.assertEqual(len(self.feature_names), 500)
        # Check that standard categories are present
        has_aac = any(name.startswith("AAC_") for name in self.feature_names)
        has_dpc = any(name.startswith("DPC_") for name in self.feature_names)
        has_svd = any(name.startswith("SVD_") for name in self.feature_names)
        self.assertTrue(has_aac, "Should contain AAC features")
        self.assertTrue(has_dpc, "Should contain DPC features")
        self.assertTrue(has_svd, "Should contain SVD features")

    def test_tree_explainer_creation_lightgbm(self):
        """Test that LightGBM TreeExplainer can be instantiated."""
        if not self.shap_installed:
            self.skipTest("SHAP not installed")
        lgb_model = self.artifacts["lightgbm"]
        explainer = get_tree_explainer(lgb_model)
        self.assertIsNotNone(explainer)

    def test_tree_explainer_creation_random_forest(self):
        """Test that Random Forest TreeExplainer can be instantiated."""
        if not self.shap_installed:
            self.skipTest("SHAP not installed")
        rf_model = self.artifacts["random_forest"]
        explainer = get_tree_explainer(rf_model)
        self.assertIsNotNone(explainer)

    def test_local_explanation_lightgbm(self):
        """Test local explanation computation on a single sample for LightGBM."""
        if not self.shap_installed:
            self.skipTest("SHAP not installed")
        lgb_model = self.artifacts["lightgbm"]
        res = explain_prediction_local(
            model=lgb_model,
            X_single=self.mock_sample,
            predicted_ec=3,
            feature_names=self.feature_names,
            top_n=5,
            model_name="LightGBM",
        )
        self.assertEqual(res["predicted_ec"], 3)
        self.assertEqual(res["class_index"], 2)
        self.assertIn("supporting_features", res)
        self.assertIn("opposing_features", res)
        self.assertEqual(len(res["raw_shap_values"]), 500)

    def test_local_explanation_random_forest(self):
        """Test local explanation computation on a single sample for Random Forest."""
        if not self.shap_installed:
            self.skipTest("SHAP not installed")
        rf_model = self.artifacts["random_forest"]
        res = explain_prediction_local(
            model=rf_model,
            X_single=self.mock_sample,
            predicted_ec=1,
            feature_names=self.feature_names,
            top_n=5,
            model_name="Random Forest",
        )
        self.assertEqual(res["predicted_ec"], 1)
        self.assertEqual(res["class_index"], 0)
        self.assertIn("supporting_features", res)
        self.assertIn("opposing_features", res)
        self.assertEqual(len(res["raw_shap_values"]), 500)

    def test_multiclass_handling_all_classes(self):
        """Test that extracting SHAP values works for all classes EC 1 to 6."""
        if not self.shap_installed:
            self.skipTest("SHAP not installed")
        lgb_model = self.artifacts["lightgbm"]
        explainer = get_tree_explainer(lgb_model)
        raw_exp = explainer(self.mock_sample)

        for c_idx in range(6):
            c_vals = extract_multiclass_shap_values(raw_exp, c_idx)
            self.assertEqual(c_vals.shape[-1], 500)

    def test_shap_does_not_modify_model(self):
        """Ensure TreeExplainer execution does not mutate model attributes or predictions."""
        if not self.shap_installed:
            self.skipTest("SHAP not installed")
        lgb_model = self.artifacts["lightgbm"]
        preds_before = lgb_model.predict_proba(self.mock_sample)

        # Run explanation
        explain_prediction_local(
            model=lgb_model,
            X_single=self.mock_sample,
            predicted_ec=2,
            feature_names=self.feature_names,
            top_n=5,
        )

        preds_after = lgb_model.predict_proba(self.mock_sample)
        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_local_waterfall_plot_rendering(self):
        """Test rendering of the local waterfall contribution figure."""
        if not self.shap_installed:
            self.skipTest("SHAP not installed")
        lgb_model = self.artifacts["lightgbm"]
        res = explain_prediction_local(
            model=lgb_model,
            X_single=self.mock_sample,
            predicted_ec=4,
            feature_names=self.feature_names,
            top_n=5,
        )
        fig = plot_local_waterfall(res, top_n=5)
        self.assertIsNotNone(fig)

    def test_existing_inference_still_works(self):
        """Verify that existing sequence prediction pipeline remains completely unaffected."""
        sample_seq = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPF"
        pred = predict_sequence(sample_seq, model_name="Stacking Ensemble", artifacts=self.artifacts)
        self.assertTrue(pred["success"])
        self.assertIn("predicted_ec", pred)
        self.assertIn("probabilities", pred)
        self.assertEqual(len(pred["probabilities"]), 6)


if __name__ == "__main__":
    unittest.main()
