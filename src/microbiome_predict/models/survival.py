"""
Model 4 — Prognostic (future-disease) modeling via survival analysis.

Forecasts the *risk of developing* a condition over time from a microbiome
profile using a Cox proportional-hazards model (``lifelines``). Because
metagenomic feature spaces are wide and sparse, the model first applies a CLR
transform, keeps the highest-variance features, and fits a penalized Cox model
for numerical stability.

``lifelines`` is an optional dependency (``pip install microbiome-predict[survival]``);
the class imports lazily and raises a clear error only if you actually try to
fit without it installed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..features import CLRTransformer

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    _HAS_LIFELINES = True
except Exception:  # pragma: no cover - environment without lifelines
    _HAS_LIFELINES = False


class PrognosticModel:
    """Cox proportional-hazards model over microbiome features.

    Parameters
    ----------
    top_k:
        Number of highest-variance (post-CLR) features to retain.
    penalizer:
        Ridge/elastic-net penalty strength for the Cox fit (stabilizes the
        wide, collinear metagenomic design matrix).
    l1_ratio:
        Elastic-net mixing parameter (0 = ridge, 1 = lasso).
    use_clr:
        Apply CLR before feature selection / fitting.
    """

    def __init__(
        self,
        top_k: int = 20,
        penalizer: float = 0.5,
        l1_ratio: float = 0.0,
        use_clr: bool = True,
    ):
        self.top_k = top_k
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.use_clr = use_clr

    def fit(self, X, durations: Sequence[float], events: Sequence[int]):
        if not _HAS_LIFELINES:
            raise ImportError(
                "PrognosticModel requires lifelines. Install with: "
                "pip install 'microbiome-predict[survival]'"
            )
        design = self._prepare(X, fit=True)
        design["__duration__"] = np.asarray(durations, dtype=float)
        design["__event__"] = np.asarray(events, dtype=int)

        self.model_ = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.model_.fit(design, duration_col="__duration__", event_col="__event__")
        self.concordance_ = float(self.model_.concordance_index_)
        return self

    def _prepare(self, X, fit: bool = False) -> pd.DataFrame:
        arr = np.asarray(X, dtype=float)
        if self.use_clr:
            arr = CLRTransformer().fit_transform(arr)
        names = _feature_names(X)
        index = X.index if hasattr(X, "index") else None

        if fit:
            variances = arr.var(axis=0)
            k = min(self.top_k, arr.shape[1])
            self.keep_idx_ = np.argsort(variances)[::-1][:k]
            self.feature_names_ = [names[i] for i in self.keep_idx_]

        selected = arr[:, self.keep_idx_]
        return pd.DataFrame(selected, columns=self.feature_names_, index=index)

    def predict_risk(self, X) -> np.ndarray:
        """Relative risk score (partial hazard); higher = earlier expected event."""
        design = self._prepare(X)
        return np.asarray(self.model_.predict_partial_hazard(design)).ravel()

    def predict_survival_function(self, X, times: Optional[Sequence[float]] = None):
        """Per-sample survival curves S(t) = P(event has not occurred by t)."""
        design = self._prepare(X)
        return self.model_.predict_survival_function(design, times=times)

    def concordance(self, X, durations, events) -> float:
        """Harrell's C-index of predicted risk vs observed outcomes."""
        if not _HAS_LIFELINES:
            raise ImportError("lifelines is required for concordance().")
        risk = self.predict_risk(X)
        # Higher risk should correspond to shorter survival -> negate for C-index.
        return float(concordance_index(np.asarray(durations, dtype=float), -risk,
                                       np.asarray(events, dtype=int)))


def _feature_names(X) -> List[str]:
    if hasattr(X, "columns"):
        return list(X.columns)
    return [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
