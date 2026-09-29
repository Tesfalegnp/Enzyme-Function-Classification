"""Inference engine for Enzyme Function Classification (EC 1–6).

Loads serialized models and transformers from `models/saved/` once,
applies identical preprocessing, feature extraction (AAC, DPC, TPC + SVD),
and feature selection, and computes predictions and class probabilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

from src.config import MODELS_DIR, STANDARD_AA, VALID_EC_CLASSES
from src.features import (
    combine_feature_matrices,
    compute_aac,
    compute_dpc,
)

# High-level official Enzyme Commission top-level classes
EC_METADATA: Dict[int, Dict[str, str]] = {
    1: {
        "code": "EC 1",
        "name": "Oxidoreductases",
        "title": "EC 1 — Oxidoreductases",
        "reaction": "Oxidation-reduction reactions (electron, hydride, or hydrogen transfer)",
        "examples": "Dehydrogenases, oxidases, peroxidases, reductases",
    },
    2: {
        "code": "EC 2",
        "name": "Transferases",
        "title": "EC 2 — Transferases",
        "reaction": "Transfer of specific functional groups (e.g., methyl, glycosyl, acyl, phospho) between donor and acceptor",
        "examples": "Kinases, transaminases, methyltransferases, polymerases",
    },
    3: {
        "code": "EC 3",
        "name": "Hydrolases",
        "title": "EC 3 — Hydrolases",
        "reaction": "Hydrolytic cleavage of covalent chemical bonds (C-O, C-N, C-C, phosphoric anhydrides) utilizing water",
        "examples": "Proteases, nucleases, esterases, phosphatases, lipases",
    },
    4: {
        "code": "EC 4",
        "name": "Lyases",
        "title": "EC 4 — Lyases",
        "reaction": "Non-hydrolytic and non-oxidative bond cleavage, often forming double bonds or new ring systems",
        "examples": "Decarboxylases, aldolases, synthases, dehydratases",
    },
    5: {
        "code": "EC 5",
        "name": "Isomerases",
        "title": "EC 5 — Isomerases",
        "reaction": "Structural, geometric, or spatial rearrangements within a single molecule",
        "examples": "Racemases, epimerases, isomerases, mutases, topoisomerases",
    },
    6: {
        "code": "EC 6",
        "name": "Ligases",
        "title": "EC 6 — Ligases",
        "reaction": "Joining of two large molecules coupled with high-energy pyrophosphate bond cleavage (ATP/GTP hydrolysis)",
        "examples": "Synthetases, DNA/RNA ligases, carboxylases",
    },
}

# Verified benchmark sample sequences from the SwissProt-EC test set
EXAMPLE_SEQUENCES: Dict[str, Dict[str, Any]] = {
    "EC 1 — Oxidoreductase (Human L-lactate dehydrogenase B-like test sample)": {
        "ec": 1,
        "name": "Oxidoreductases",
        "source": "SwissProt-EC held-out test partition",
        "sequence": (
            "MARAAPLLAALTALLAAAAAGGDAPPGKIAVVGAGIGGSAVAHFLQQHFGPRVQIDVYEKGTVGGRLATISVNKQHYES"
            "GAASFHSLSLHMQDFVKLLGLRHRREVVGRSAIFGGEHFMLEETDWYLLNLFRLWWHYGISFLRLQMWVEEVMEKFMRI"
            "YKYQAHGYAFSGVEELLYSLGESTFVNMTQHSVAESLLQVGVTQRFIDDVVSAVLRASYGQSAAMPAFAGAMSLAGAQG"
            "SLWSVEGGNKLVCSGLLKLTKANVIHATVTSVTLHSTEGKALYQVAYENEVGNSSDFYDIVVIATPLHLDNSSSNLTFA"
            "GFHPPIDDVQGSFQPTVVSLVHGYLNSSYFGFPDPKLFPFANILTTDFPSFFCTLDNICPVNISASFRRKQPQEAAVWR"
            "VQSPKPLFRTQLKTLFRSYYSVQTAEWQAHPLYGSRPTLPRFALHDQLFYLNALEWAASSVEVMAVAAKNVALLAYNRW"
            "YQDLDKIDQKDLMHKVKTEL"
        ),
    },
    "EC 3 — Hydrolase (Bacterial esterase/hydrolase test sample)": {
        "ec": 3,
        "name": "Hydrolases",
        "source": "SwissProt-EC held-out test partition",
        "sequence": (
            "MVRRLWRRIAGWLAACVAILCAFPLHAATAGPGAWSSQQTWAADSVNGGNLTGYFYWPASQPTTPNGKRALVLVLHGCV"
            "QTASGDVIDNANGAGFNWKSVADQYGAVILAPNATGNVYSNHCWDYANASPSRTAGHVGVLLDLVNRFVTNSQYAIDPN"
            "QVYVAGLSSGGGMTMVLGCIAPDIFAGIGINAGPPPGTTTAQIGYVPSGFTATTAANKCNAWAGSNAGKFSTQIAGAVW"
            "GTSDYTVAQAYGPMDAAAMRLVYGGNFTQGSQVSISGGGTNTPYTDSNGKVRTHEISVSGMAHAWPAGTGGDNTNYVDA"
            "THINYPVFVMDYWVKNNLRAGSGTGQAGSAPTGLAVTATTSTSVSLSWNAVANASSYGVYRNGSKVGSATATAYTDSGL"
            "IAGTTYSYTVTAVDPTAGESQPSAAVSATTKSAFTCTATTASNYAHVQAGRAHDSGGIAYANGSNQSMGLDNLFYTSTL"
            "AQTAAGYYIVGNCP"
        ),
    },
    "EC 5 — Isomerase (Triosephosphate isomerase test sample)": {
        "ec": 5,
        "name": "Isomerases",
        "source": "SwissProt-EC held-out test partition",
        "sequence": (
            "MRRKIVVGNWKMNNSVAESVQLATDVLAALGEGFSGCEVGIAPTYLALDATEKVIAESEVQLVAQNCHYENDGAFTGEV"
            "SARMILAVGCSSVIIGHSERRQYFGETNATVNLRIKKALSEGLNVILCVGETLAERESGVMETVISSQVREGLDGIIDI"
            "SAIVIAYEPVWAIGTGKTASSAQAEEVHLFIRTLVTGLYGQTASEKVRIQYGGSVKPSNAAELFAMPNIDGGLIGGASL"
            "NADDFAAIVKAASV"
        ),
    },
}

_GLOBAL_ARTIFACTS: Optional[Dict[str, Any]] = None


def clean_and_validate_sequence(raw_text: str) -> Tuple[bool, str, Optional[str]]:
    """Clean and validate an amino-acid protein sequence.

    Supports:
    - Single-line sequences
    - Multi-line sequences
    - FASTA format (strips header starting with '>')
    - Ignores internal whitespaces, numbers, and carriage returns
    - Checks standard 20 amino-acid vocabulary

    Parameters
    ----------
    raw_text : str
        Input string from user or FASTA file.

    Returns
    -------
    Tuple[bool, str, Optional[str]]
        (is_valid, cleaned_sequence, error_message)
    """
    if not raw_text or not isinstance(raw_text, str):
        return False, "", "Input is empty. Please enter a valid protein sequence."

    lines = raw_text.strip().splitlines()
    seq_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">") or stripped.startswith(";"):
            # FASTA header or comment
            continue
        # Remove any whitespace, tabs, or line numbering
        cleaned_line = "".join(ch for ch in stripped if ch.isalpha())
        if cleaned_line:
            seq_lines.append(cleaned_line.upper())

    sequence = "".join(seq_lines)

    if not sequence:
        return False, "", "No protein sequence characters found after stripping headers and formatting."

    if len(sequence) < 10:
        return False, sequence, f"Sequence length ({len(sequence)} aa) is too short. Minimum recommended length is 10 aa."

    standard_set = set(STANDARD_AA)
    non_standard = sorted(list(set(sequence) - standard_set))

    if non_standard:
        chars_str = ", ".join(non_standard)
        return (
            False,
            sequence,
            f"Sequence contains non-standard amino-acid character(s): {chars_str}. "
            f"The model supports the standard 20 amino acids: {''.join(STANDARD_AA)}.",
        )

    return True, sequence, None


def load_inference_artifacts(models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load serialized transformers, preprocessors, and models into memory.

    Caches artifacts globally to guarantee fast, zero-training inference.
    """
    global _GLOBAL_ARTIFACTS
    if _GLOBAL_ARTIFACTS is not None:
        return _GLOBAL_ARTIFACTS

    target_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    if not target_dir.exists():
        raise FileNotFoundError(f"Saved models directory does not exist: {target_dir}")

    required_artifacts = [
        "tripeptide_vectorizer.pkl",
        "truncated_svd.pkl",
        "select_k_best.pkl",
        "standard_scaler.pkl",
        "random_forest.pkl",
        "lightgbm.pkl",
        "svm.pkl",
        "calibrated_svm.pkl",
        "stacking_meta_learner.pkl",
    ]

    for fname in required_artifacts:
        fpath = target_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Required model artifact missing: {fpath}")

    artifacts = {
        "vectorizer": joblib.load(target_dir / "tripeptide_vectorizer.pkl"),
        "svd": joblib.load(target_dir / "truncated_svd.pkl"),
        "selector": joblib.load(target_dir / "select_k_best.pkl"),
        "scaler": joblib.load(target_dir / "standard_scaler.pkl"),
        "random_forest": joblib.load(target_dir / "random_forest.pkl"),
        "lightgbm": joblib.load(target_dir / "lightgbm.pkl"),
        "svm": joblib.load(target_dir / "svm.pkl"),
        "calibrated_svm": joblib.load(target_dir / "calibrated_svm.pkl"),
        "stacking_meta_learner": joblib.load(target_dir / "stacking_meta_learner.pkl"),
    }

    _GLOBAL_ARTIFACTS = artifacts
    return artifacts


def extract_sequence_features(
    sequence: str, artifacts: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a single cleaned protein sequence into the 500D model feature space.

    Applies the exact same steps used during training:
    1. AAC (20D)
    2. DPC (400D)
    3. TPC sparse count transformation (via training-fit CountVectorizer)
    4. TruncatedSVD transformation (via training-fit SVD to 128D)
    5. Feature combination (548D)
    6. SelectKBest transformation (via training-fit ANOVA F-selector to 500D)
    7. StandardScaler transformation for SVM (via training-fit scaler)

    Parameters
    ----------
    sequence : str
        Cleaned protein sequence containing only standard 20 amino acids.
    artifacts : Optional[Dict[str, Any]]
        Preloaded artifacts dictionary (loaded if None).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_selected, X_scaled) of shape (1, 500) each.
    """
    if artifacts is None:
        artifacts = load_inference_artifacts()

    # 1. AAC (1, 20)
    aac = compute_aac([sequence])

    # 2. DPC (1, 400)
    dpc = compute_dpc([sequence])

    # 3. TPC sparse (1, V)
    tpc = artifacts["vectorizer"].transform([sequence])

    # 4. TruncatedSVD (1, 128)
    emb = artifacts["svd"].transform(tpc).astype(np.float32)

    # 5. Combined features (1, 548)
    comb = combine_feature_matrices(aac, dpc, emb)

    # 6. SelectKBest (1, 500)
    X_selected = artifacts["selector"].transform(comb).astype(np.float32)

    # 7. Scaled features for SVM (1, 500)
    X_scaled = artifacts["scaler"].transform(X_selected).astype(np.float32)

    return X_selected, X_scaled


def predict_sequence(
    raw_sequence: str,
    model_name: str = "Stacking Ensemble",
    artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """End-to-end prediction from raw sequence to EC 1-6 classification.

    Parameters
    ----------
    raw_sequence : str
        Raw input string (FASTA or plain text).
    model_name : str
        Selected model. Supported:
        - "Stacking Ensemble" (Recommended, highest test accuracy)
        - "Random Forest" (Highest test macro F1)
        - "LightGBM"
        - "Soft Voting Ensemble"
        - "Calibrated Linear SVM"
    artifacts : Optional[Dict[str, Any]]
        Preloaded artifacts dictionary.

    Returns
    -------
    Dict[str, Any]
        Complete inference output with predicted EC, metadata, and probability distribution.
    """
    is_valid, clean_seq, err_msg = clean_and_validate_sequence(raw_sequence)
    if not is_valid:
        return {
            "success": False,
            "error": err_msg,
            "cleaned_sequence": clean_seq,
        }

    if artifacts is None:
        artifacts = load_inference_artifacts()

    X_sel, X_sc = extract_sequence_features(clean_seq, artifacts=artifacts)

    # Compute base model outputs
    p_rf = artifacts["random_forest"].predict_proba(X_sel)
    p_lgb = artifacts["lightgbm"].predict_proba(X_sel)
    d_svm = artifacts["svm"].decision_function(X_sc)
    p_svm_cal = artifacts["calibrated_svm"].predict_proba(X_sc)

    # Base predictions
    base_preds = {
        "Random Forest": int(artifacts["random_forest"].predict(X_sel)[0]),
        "LightGBM": int(artifacts["lightgbm"].predict(X_sel)[0]),
        "Linear SVM": int(artifacts["svm"].predict(X_sc)[0]),
    }

    # Meta-feature for Stacking
    meta_X = np.hstack([p_rf, p_lgb, d_svm])

    if model_name == "Stacking Ensemble":
        meta_learner = artifacts["stacking_meta_learner"]
        pred_class = int(meta_learner.predict(meta_X)[0])
        probs_array = meta_learner.predict_proba(meta_X)[0]
        model_desc = "Stacking Ensemble with Logistic Regression meta-learner over RF, LightGBM, and SVM outputs"
    elif model_name == "Random Forest":
        pred_class = base_preds["Random Forest"]
        probs_array = p_rf[0]
        model_desc = "Random Forest (100 trees, depth 20, class-balanced)"
    elif model_name == "LightGBM":
        pred_class = base_preds["LightGBM"]
        probs_array = p_lgb[0]
        model_desc = "LightGBM Classifier (100 trees, 31 leaves, feature fraction 0.5)"
    elif model_name == "Soft Voting Ensemble":
        probs_array = (p_rf[0] + p_lgb[0] + p_svm_cal[0]) / 3.0
        pred_class = int(np.argmax(probs_array) + 1)
        model_desc = "Equal-weighted soft voting across RF, LightGBM, and Platt-calibrated LinearSVC"
    elif model_name == "Calibrated Linear SVM":
        pred_class = int(artifacts["calibrated_svm"].predict(X_sc)[0])
        probs_array = p_svm_cal[0]
        model_desc = "LinearSVC with Platt scaling (CalibratedClassifierCV, sigmoid)"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    probabilities = {
        cls: float(probs_array[idx]) for idx, cls in enumerate(VALID_EC_CLASSES)
    }

    ec_info = EC_METADATA.get(pred_class, {
        "code": f"EC {pred_class}",
        "name": "Unknown",
        "title": f"EC {pred_class}",
        "reaction": "",
        "examples": "",
    })

    # Amino acid composition summary
    seq_len = len(clean_seq)
    aa_counts = {aa: clean_seq.count(aa) for aa in STANDARD_AA}
    top_aa = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_aa_summary = [
        {"amino_acid": aa, "count": count, "frequency": count / seq_len}
        for aa, count in top_aa
    ]

    return {
        "success": True,
        "predicted_ec": pred_class,
        "ec_code": ec_info["code"],
        "ec_name": ec_info["name"],
        "ec_title": ec_info["title"],
        "ec_reaction": ec_info["reaction"],
        "ec_examples": ec_info["examples"],
        "confidence": probabilities[pred_class],
        "probabilities": probabilities,
        "model_name": model_name,
        "model_description": model_desc,
        "sequence_length": seq_len,
        "cleaned_sequence": clean_seq,
        "base_model_predictions": base_preds,
        "top_amino_acids": top_aa_summary,
    }
