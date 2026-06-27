"""
Model 3 — Ensemble machine learning for specific-disease classification.

A soft-voting ensemble that averages calibrated class probabilities from four
complementary learners:

* Random Forest          — robust to noise, captures non-linear interactions
* Support Vector Machine — strong margins in high-dimensional spaces
* Gradient Boosting      — sequential error correction
* XGBoost (if installed) — regularized, high-performance boosting

Inputs are handled compositionally (optional prevalence filter -> CLR ->
standardization) before the voting layer. Keeping the individual fitted members
accessible (``member_proba``) lets the uncertainty module quantify
*disagreement between models* as a per-prediction confidence signal.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from ..features import CLRTransformer, PrevalenceFilter

try:  # XGBoost is a hard dependency in pyproject, but degrade gracefully.
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # pragma: no cover - environment without xgboost
    _HAS_XGB = False


def _build_members(random_state: int) -> List[Tuple[str, BaseEstimator]]:
    members: List[Tuple[str, BaseEstimator]] = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(random_state=random_state),
        ),
    ]
    if _HAS_XGB:
        members.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )
    return members


class EnsembleDiseaseClassifier(BaseEstimator, ClassifierMixin):
    """Soft-voting microbiome disease classifier.

    Parameters
    ----------
    min_prevalence:
        Drop taxa present in fewer than this fraction of training samples.
    use_clr:
        Apply the centered-log-ratio transform (recommended for compositional
        data). If ``False``, raw values are only standardized.
    random_state:
        Seed propagated to every member for reproducibility.
    """

    def __init__(
        self,
        min_prevalence: float = 0.1,
        use_clr: bool = True,
        random_state: int = 0,
    ):
        self.min_prevalence = min_prevalence
        self.use_clr = use_clr
        self.random_state = random_state

    # -- construction ---------------------------------------------------------
    def _build_pipeline(self) -> Pipeline:
        steps: List[Tuple[str, BaseEstimator]] = [
            ("prevalence", PrevalenceFilter(self.min_prevalence))
        ]
        if self.use_clr:
            steps.append(("clr", CLRTransformer()))
        steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "ensemble",
                VotingClassifier(
                    estimators=_build_members(self.random_state),
                    voting="soft",
                ),
            )
        )
        return Pipeline(steps)

    # -- sklearn API ----------------------------------------------------------
    def fit(self, X, y):
        self.feature_names_ = _feature_names(X)
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(np.asarray(y))
        self.classes_ = self._label_encoder.classes_
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(_as_float(X), y_encoded)
        return self

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(_as_float(X))

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

    def decision_features(self, X):
        """Return X after the pre-ensemble transforms (prevalence/CLR/scaler)."""
        transformed = _as_float(X)
        for _, step in self.pipeline_.steps[:-1]:
            transformed = step.transform(transformed)
        return transformed

    def member_proba(self, X) -> np.ndarray:
        """Per-member class probabilities, shape ``(n_members, n_samples, n_classes)``.

        Used by the uncertainty module: the spread across members at a given
        sample is a direct, model-derived measure of predictive confidence
        (analogous to a pLDDT-style per-prediction reliability score).
        """
        transformed = self.decision_features(X)
        ensemble: VotingClassifier = self.pipeline_.named_steps["ensemble"]
        probas = [
            est.predict_proba(transformed)
            for est in ensemble.named_estimators_.values()
        ]
        return np.stack(probas, axis=0)

    @property
    def member_names_(self) -> List[str]:
        ensemble: VotingClassifier = self.pipeline_.named_steps["ensemble"]
        return list(ensemble.named_estimators_.keys())


def _feature_names(X) -> List[str]:
    if hasattr(X, "columns"):
        return list(X.columns)
    return [f"feature_{i}" for i in range(np.asarray(X).shape[1])]


def _as_float(X) -> np.ndarray:
    return np.asarray(X, dtype=float)
