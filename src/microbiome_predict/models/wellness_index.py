"""
Model 2 — Wellness index (GMWI2-style health-status indexing).

The published Gut Microbiome Wellness Index 2 (GMWI2) scores a sample by
contrasting the abundance of taxa that are characteristically enriched in
*healthy* versus *non-healthy* individuals, then thresholds that score to call
"healthy", "diseased", or (in the reject band) "indeterminate".

This implementation reproduces that *transparent, interpretable* structure in a
data-driven way: during ``fit`` it learns which taxa are most discriminating
from the supplied healthy/non-healthy labels, and the score for a new sample is
a simple, auditable log-ratio of those two taxa sets. It is GMWI2-*inspired* and
does not ship the exact published coefficient list — point it at your own
curated taxa sets via ``health_plus_``/``health_minus_`` if you need the
canonical index.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.base import BaseEstimator

HEALTHY = "HEALTHY"
DISEASED = "DISEASED"
INDETERMINATE = "INDETERMINATE"


class WellnessIndex(BaseEstimator):
    """Interpretable healthy-vs-diseased microbiome index.

    Parameters
    ----------
    n_taxa:
        Number of taxa to use on each side (health-positive / health-negative).
    healthy_label:
        The label value in ``y`` that denotes a healthy sample.
    reject_low, reject_high:
        Scores in the open interval ``(reject_low, reject_high)`` are returned
        as ``INDETERMINATE`` (the reject option). Scores are centered on the
        training median, so 0 is the decision boundary.
    pseudocount:
        Added to abundances before the log-ratio to keep zeros finite.
    """

    def __init__(
        self,
        n_taxa: int = 20,
        healthy_label=0,
        reject_low: float = -0.25,
        reject_high: float = 0.25,
        pseudocount: float = 1e-6,
    ):
        self.n_taxa = n_taxa
        self.healthy_label = healthy_label
        self.reject_low = reject_low
        self.reject_high = reject_high
        self.pseudocount = pseudocount

    def fit(self, X, y):
        arr = np.asarray(X, dtype=float)
        names = _feature_names(X)
        y = np.asarray(y)

        healthy_mask = y == self.healthy_label
        if healthy_mask.all() or (~healthy_mask).all():
            raise ValueError(
                "WellnessIndex requires both healthy and non-healthy samples "
                f"(healthy_label={self.healthy_label!r})."
            )

        mean_healthy = arr[healthy_mask].mean(axis=0)
        mean_diseased = arr[~healthy_mask].mean(axis=0)
        diff = mean_healthy - mean_diseased

        n = min(self.n_taxa, arr.shape[1] // 2 or 1)
        order = np.argsort(diff)
        self.health_minus_idx_ = order[:n]            # enriched in diseased
        self.health_plus_idx_ = order[::-1][:n]       # enriched in healthy
        self.health_plus_ = [names[i] for i in self.health_plus_idx_]
        self.health_minus_ = [names[i] for i in self.health_minus_idx_]
        self.feature_names_ = names

        # Center on the training median so 0 is the natural decision boundary.
        self.center_ = float(np.median(self._raw_score(arr)))
        return self

    def _raw_score(self, arr: np.ndarray) -> np.ndarray:
        plus = arr[:, self.health_plus_idx_].mean(axis=1)
        minus = arr[:, self.health_minus_idx_].mean(axis=1)
        return np.log10((plus + self.pseudocount) / (minus + self.pseudocount))

    def score_samples(self, X) -> np.ndarray:
        """Centered wellness score; positive = healthier, negative = more diseased."""
        arr = np.asarray(X, dtype=float)
        return self._raw_score(arr) - getattr(self, "center_", 0.0)

    def predict(self, X) -> np.ndarray:
        scores = self.score_samples(X)
        out = np.full(len(scores), INDETERMINATE, dtype=object)
        out[scores >= self.reject_high] = HEALTHY
        out[scores <= self.reject_low] = DISEASED
        return out


def _feature_names(X) -> List[str]:
    if hasattr(X, "columns"):
        return list(X.columns)
    return [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
