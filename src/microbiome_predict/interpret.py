"""
Interpretability and actionable insight (spec Sections 4 & 6).

* ``global_importance`` — which taxa drive the model overall (permutation
  importance over the full fitted pipeline; model-agnostic and leakage-free).
* ``local_explanation`` — why *this* sample got *this* prediction. Uses SHAP if
  it is installed, otherwise a model-agnostic *occlusion* fallback (replace each
  feature with a background reference and measure the change in the predicted
  class probability). Positive values pushed the prediction up.

Keeping a non-SHAP fallback means the explanations always work, even in the
slim core install.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def has_shap() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except Exception:
        return False


def global_importance(
    clf,
    X,
    y,
    n_repeats: int = 5,
    random_state: int = 0,
) -> pd.Series:
    """Permutation feature importance, returned as a sorted ``Series``.

    Works on any fitted scikit-learn-compatible estimator (including the
    ensemble pipeline). Importance = mean drop in score when a feature is
    shuffled.
    """
    result = permutation_importance(
        clf,
        np.asarray(X, dtype=float),
        np.asarray(y),
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    names = _feature_names(clf, X)
    series = pd.Series(result.importances_mean, index=names, name="importance")
    return series.sort_values(ascending=False)


def local_explanation(
    clf,
    x_row,
    background_reference: Optional[np.ndarray] = None,
    target_class=None,
    k: int = 15,
    max_features_evaluated: int = 60,
) -> pd.Series:
    """Per-sample feature attribution for a single prediction.

    If SHAP is available and a background set is provided it is used; otherwise
    an occlusion analysis estimates each feature's contribution to the predicted
    class probability. Returns the top ``k`` features by absolute contribution.
    """
    x = np.asarray(x_row, dtype=float).reshape(1, -1)
    names = _feature_names(clf, x_row)
    base = clf.predict_proba(x)[0]

    if target_class is None:
        target_idx = int(np.argmax(base))
    else:
        target_idx = int(list(clf.classes_).index(target_class))
    base_p = base[target_idx]

    if has_shap() and background_reference is not None:
        contrib = _shap_local(clf, x, background_reference, target_idx, names)
        if contrib is not None:
            return _top_k(contrib, k)

    # --- occlusion fallback --------------------------------------------------
    if background_reference is None:
        ref = np.zeros_like(x)
    else:
        ref = np.asarray(background_reference, dtype=float).reshape(1, -1)

    # Only evaluate the features that differ most from the reference — the rest
    # contribute ~nothing and this bounds the number of model calls.
    delta = np.abs(x[0] - ref[0])
    candidate_idx = np.argsort(delta)[::-1][:max_features_evaluated]

    contrib = np.zeros(x.shape[1])
    for j in candidate_idx:
        perturbed = x.copy()
        perturbed[0, j] = ref[0, j]
        p = clf.predict_proba(perturbed)[0][target_idx]
        contrib[j] = base_p - p  # positive => feature raised the predicted prob

    series = pd.Series(contrib, index=names, name="contribution")
    return _top_k(series, k)


def _shap_local(clf, x, background, target_idx, names) -> Optional[pd.Series]:
    try:
        import shap

        background = np.asarray(background, dtype=float)
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        values = explainer.shap_values(x, nsamples=100, silent=True)
        if isinstance(values, list):
            vals = np.asarray(values[target_idx]).ravel()
        else:  # newer SHAP returns (n, features, classes)
            arr = np.asarray(values)
            vals = arr[0, :, target_idx] if arr.ndim == 3 else arr.ravel()
        return pd.Series(vals, index=names, name="contribution")
    except Exception:
        return None


def _top_k(series: pd.Series, k: int) -> pd.Series:
    order = series.abs().sort_values(ascending=False).index
    return series.reindex(order).head(k)


def _feature_names(clf, X) -> List[str]:
    names = getattr(clf, "feature_names_", None)
    if names is not None:
        return list(names)
    if hasattr(X, "columns"):
        return list(X.columns)
    return [f"feature_{i}" for i in range(np.asarray(X).reshape(1, -1).shape[1])]
