"""
Feature transforms for compositional microbiome data.

Microbiome abundance vectors are *compositional* (they live on the simplex —
each sample sums to a constant), so naively feeding raw relative abundances to
distance/variance-based learners is statistically inappropriate. The
centered-log-ratio (CLR) transform maps compositions into ordinary Euclidean
space and is the standard remedy.

All transformers here are scikit-learn compatible so they slot directly into a
``Pipeline``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PrevalenceFilter(BaseEstimator, TransformerMixin):
    """Drop features (taxa) observed in fewer than ``min_prevalence`` of samples.

    Prevalence is learned on the training data only and then applied
    identically at predict time, avoiding train/test leakage.
    """

    def __init__(self, min_prevalence: float = 0.0):
        self.min_prevalence = min_prevalence

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        prevalence = (arr > 0).mean(axis=0)
        keep = prevalence >= self.min_prevalence
        # Never collapse to zero features — fall back to keeping everything.
        if not keep.any():
            keep = np.ones(arr.shape[1], dtype=bool)
        self.keep_ = keep
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return arr[:, self.keep_]


class RelativeAbundance(BaseEstimator, TransformerMixin):
    """Row-normalize so each sample sums to 1.0 (a no-op fit)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        totals = arr.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return arr / totals


class CLRTransformer(BaseEstimator, TransformerMixin):
    """Centered log-ratio transform with a pseudocount for zeros.

    For a composition x, CLR(x)_i = log(x_i) - mean_j log(x_j). A small
    ``pseudocount`` is added first so that zeros (very common in sparse
    metagenomic profiles) don't produce -inf.
    """

    def __init__(self, pseudocount: float = 1e-6):
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float) + self.pseudocount
        log_arr = np.log(arr)
        return log_arr - log_arr.mean(axis=1, keepdims=True)
