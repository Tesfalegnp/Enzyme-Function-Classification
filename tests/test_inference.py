"""Unit tests for the inference engine and sequence validation."""

import unittest
import numpy as np

from src.inference import (
    clean_and_validate_sequence,
    extract_sequence_features,
    load_inference_artifacts,
    predict_sequence,
    EXAMPLE_SEQUENCES,
    EC_METADATA,
)


class TestInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Preload artifacts once for all tests
        cls.artifacts = load_inference_artifacts()
        cls.test_seq = (
            "MARAAPLLAALTALLAAAAAGGDAPPGKIAVVGAGIGGSAVAHFLQQHFGPRVQIDVYEKGTVGGRLATISVNKQHYES"
            "GAASFHSLSLHMQDFVKLLGLRHRREVVGRSAIFGGEHFMLEETDWYLLNLFRLWWHYGISFLRLQMWVEEVMEKFMRI"
            "YKYQAHGYAFSGVEELLYSLGESTFVNMTQHSVAESLLQVGVTQRFIDDVVSAVLRASYGQSAAMPAFAGAMSLAGAQG"
            "SLWSVEGGNKLVCSGLLKLTKANVIHATVTSVTLHSTEGKALYQVAYENEVGNSSDFYDIVVIATPLHLDNSSSNLTFA"
            "GFHPPIDDVQGSFQPTVVSLVHGYLNSSYFGFPDPKLFPFANILTTDFPSFFCTLDNICPVNISASFRRKQPQEAAVWR"
            "VQSPKPLFRTQLKTLFRSYYSVQTAEWQAHPLYGSRPTLPRFALHDQLFYLNALEWAASSVEVMAVAAKNVALLAYNRW"
            "YQDLDKIDQKDLMHKVKTEL"
        )

    def test_clean_and_validate_valid_plain(self):
        is_valid, clean_seq, err = clean_and_validate_sequence("ACDEFGHIKLMNPQRSTVWY")
        self.assertTrue(is_valid)
        self.assertEqual(clean_seq, "ACDEFGHIKLMNPQRSTVWY")
        self.assertIsNone(err)

    def test_clean_and_validate_fasta_format(self):
        fasta = (
            ">sp|P00338|LDHA_HUMAN L-lactate dehydrogenase A chain\n"
            "MATLKDQLIQ NLLKEEHVPN KITIIVGVGAV\n"
            "GMACAISILM KDLADELALV DVMEDKLKGE\n"
        )
        is_valid, clean_seq, err = clean_and_validate_sequence(fasta)
        self.assertTrue(is_valid)
        self.assertNotIn(">", clean_seq)
        self.assertNotIn(" ", clean_seq)
        self.assertNotIn("\n", clean_seq)
        self.assertEqual(clean_seq, "MATLKDQLIQNLLKEEHVPNKITIIVGVGAVGMACAISILMKDLADELALVDVMEDKLKGE")

    def test_clean_and_validate_empty_input(self):
        is_valid, clean_seq, err = clean_and_validate_sequence("")
        self.assertFalse(is_valid)
        self.assertIn("empty", err.lower())

        is_valid, clean_seq, err = clean_and_validate_sequence("   \n\t  ")
        self.assertFalse(is_valid)

    def test_clean_and_validate_too_short(self):
        is_valid, clean_seq, err = clean_and_validate_sequence("ACDEF")
        self.assertFalse(is_valid)
        self.assertIn("too short", err.lower())

    def test_clean_and_validate_non_standard_amino_acids(self):
        # 'B', 'Z', 'X', 'U' are non-standard in 20-AA alphabet
        is_valid, clean_seq, err = clean_and_validate_sequence("ACDEFGHIKLMNPQRSTVWYBZ")
        self.assertFalse(is_valid)
        self.assertIn("non-standard", err.lower())
        self.assertIn("B", err)
        self.assertIn("Z", err)

    def test_extract_sequence_features_shape(self):
        X_sel, X_sc = extract_sequence_features(self.test_seq, artifacts=self.artifacts)
        self.assertEqual(X_sel.shape, (1, 500))
        self.assertEqual(X_sc.shape, (1, 500))
        self.assertEqual(X_sel.dtype, np.float32)
        self.assertEqual(X_sc.dtype, np.float32)

    def test_predict_sequence_stacking(self):
        res = predict_sequence(self.test_seq, model_name="Stacking Ensemble", artifacts=self.artifacts)
        self.assertTrue(res["success"])
        self.assertEqual(res["predicted_ec"], 1)
        self.assertEqual(res["ec_name"], "Oxidoreductases")
        self.assertIn("probabilities", res)
        self.assertEqual(len(res["probabilities"]), 6)
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)
        self.assertGreater(res["confidence"], 0.5)

    def test_predict_sequence_other_models(self):
        for model in ["Random Forest", "LightGBM", "Soft Voting Ensemble"]:
            res = predict_sequence(self.test_seq, model_name=model, artifacts=self.artifacts)
            self.assertTrue(res["success"])
            self.assertIn(res["predicted_ec"], range(1, 7))
            self.assertEqual(len(res["probabilities"]), 6)
            self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=3)

    def test_predict_invalid_sequence_graceful_failure(self):
        res = predict_sequence("INVALID_AMINO_ACIDS_XYZ", artifacts=self.artifacts)
        self.assertFalse(res["success"])
        self.assertIn("error", res)

    def test_example_sequences_present_and_valid(self):
        self.assertGreaterEqual(len(EXAMPLE_SEQUENCES), 3)
        for name, data in EXAMPLE_SEQUENCES.items():
            self.assertIn("ec", data)
            self.assertIn("sequence", data)
            is_valid, _, _ = clean_and_validate_sequence(data["sequence"])
            self.assertTrue(is_valid, f"Example {name} failed validation")


if __name__ == "__main__":
    unittest.main()
