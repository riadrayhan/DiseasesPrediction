"""
Model 1 — Interpretable AI for host-status prediction (MMETHANE-style).

The spec calls for MMETHANE, an external, separately-licensed interpretable
deep-learning model whose key value is producing *human-readable rules*. Rather
than vendor a heavyweight external dependency, this module provides:

1. ``InterpretableRuleClassifier`` — a self-contained, fully interpretable
   classifier (a shallow decision tree over relative abundances) that emits
   plain-English IF/THEN rules such as
   ``IF Fusobacterium_nucleatum > 0.80% AND Bacteroides_fragilis <= 0.20%
   THEN colorectal_cancer``; and
2. ``MMETHANEAdapter`` — a thin protocol/wrapper so a real MMETHANE model (or
   any model exposing ``fit``/``predict_proba``/``rules``) can be dropped in
   without changing the rest of the pipeline.

Decision thresholds are kept in *relative-abundance* space (not CLR) precisely
so the rules read in clinically meaningful units (percent of the community).
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from ..features import PrevalenceFilter, RelativeAbundance


class InterpretableRuleClassifier(BaseEstimator, ClassifierMixin):
    """Glass-box host-status classifier that exposes human-readable rules.

    Parameters
    ----------
    max_depth:
        Maximum tree depth (smaller = simpler, more readable rules).
    min_prevalence:
        Drop taxa seen in fewer than this fraction of training samples.
    min_samples_leaf:
        Minimum samples per leaf (regularizes the rules).
    random_state:
        Reproducibility seed.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_prevalence: float = 0.1,
        min_samples_leaf: int = 5,
        random_state: int = 0,
    ):
        self.max_depth = max_depth
        self.min_prevalence = min_prevalence
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        self.feature_names_ = _feature_names(X)
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(np.asarray(y))
        self.classes_ = self._label_encoder.classes_

        self._prevalence = PrevalenceFilter(self.min_prevalence).fit(X)
        self.kept_features_ = [
            name
            for name, keep in zip(self.feature_names_, self._prevalence.keep_)
            if keep
        ]
        self._relative = RelativeAbundance()
        transformed = self._transform(X)

        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_state,
        ).fit(transformed, y_encoded)
        return self

    def _transform(self, X):
        return self._relative.transform(self._prevalence.transform(X))

    def predict_proba(self, X):
        return self.tree_.predict_proba(self._transform(X))

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

    # -- interpretability -----------------------------------------------------
    def rules(self) -> List[str]:
        """Return the decision paths as human-readable IF/THEN rule strings."""
        tree = self.tree_.tree_
        feature_names = self.kept_features_
        class_labels = self.classes_
        rules: List[str] = []

        def recurse(node: int, conditions: List[str]):
            if tree.children_left[node] == tree.children_right[node]:  # leaf
                counts = tree.value[node][0]
                total = counts.sum()
                cls = class_labels[int(np.argmax(counts))]
                purity = counts.max() / total if total else 0.0
                premise = " AND ".join(conditions) if conditions else "always"
                rules.append(
                    f"IF {premise} THEN {cls} "
                    f"(n={int(total)}, purity={purity:.0%})"
                )
                return
            name = feature_names[tree.feature[node]]
            # Thresholds are relative abundances -> render as percentages.
            threshold = tree.threshold[node] * 100
            recurse(
                tree.children_left[node],
                conditions + [f"{name} <= {threshold:.2f}%"],
            )
            recurse(
                tree.children_right[node],
                conditions + [f"{name} > {threshold:.2f}%"],
            )

        recurse(0, [])
        return rules

    def rules_text(self) -> str:
        return "\n".join(self.rules())


class MMETHANEAdapter:
    """Wrap an external MMETHANE-style model behind the common interface.

    The wrapped object must implement ``fit(X, y)`` and ``predict_proba(X)``;
    if it also exposes ``rules()`` those are surfaced, otherwise a placeholder
    is returned. This lets a real MMETHANE deployment slot in wherever
    :class:`InterpretableRuleClassifier` is used.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = getattr(self.model, "classes_", None)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        idx = np.argmax(self.predict_proba(X), axis=1)
        return np.asarray(self.classes_)[idx]

    def rules(self) -> List[str]:
        if hasattr(self.model, "rules"):
            return list(self.model.rules())
        return ["(wrapped model does not expose human-readable rules)"]


def _feature_names(X) -> List[str]:
    if hasattr(X, "columns"):
        return list(X.columns)
    return [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
