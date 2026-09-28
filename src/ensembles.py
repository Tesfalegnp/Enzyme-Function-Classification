"""Ensemble classifiers: Hard Voting, Calibrated Soft Voting, and Stacking."""

from typing import Any, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from src.config import N_JOBS, RANDOM_SEED


def build_hard_voting_classifier(
    rf_model: Any,
    lgb_model: Any,
    svm_model: Any,
) -> VotingClassifier:
    """Construct Hard Voting (majority rule) ensemble from base models."""
    return VotingClassifier(
        estimators=[
            ("rf", rf_model),
            ("lgb", lgb_model),
            ("svm", svm_model),
        ],
        voting="hard",
        n_jobs=N_JOBS,
    )


def build_soft_voting_classifier(
    rf_model: Any,
    lgb_model: Any,
    svm_model: Any,
) -> VotingClassifier:
    """Construct Soft Voting ensemble using probability estimates.

    If the SVM model is a raw LinearSVC lacking predict_proba, it is wrapped in
    CalibratedClassifierCV so true posterior probabilities are combined.
    """
    if not hasattr(svm_model, "predict_proba"):
        calibrated_svm = CalibratedClassifierCV(
            estimator=svm_model,
            method="sigmoid",
            cv=3,
            n_jobs=N_JOBS,
        )
    else:
        calibrated_svm = svm_model

    return VotingClassifier(
        estimators=[
            ("rf", rf_model),
            ("lgb", lgb_model),
            ("svm", calibrated_svm),
        ],
        voting="soft",
        n_jobs=N_JOBS,
    )


def build_stacking_classifier(
    rf_model: Any,
    lgb_model: Any,
    svm_model: Any,
    cv: int = 3,
    random_state: int = RANDOM_SEED,
) -> StackingClassifier:
    """Construct Stacking Classifier with Logistic Regression meta-learner.

    cv=3 is used for internal cross-validation to maintain high out-of-fold generalization
    while preserving local CPU efficiency.
    """
    meta_learner = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )

    return StackingClassifier(
        estimators=[
            ("rf", rf_model),
            ("lgb", lgb_model),
            ("svm", svm_model),
        ],
        final_estimator=meta_learner,
        cv=cv,
        n_jobs=N_JOBS,
    )
