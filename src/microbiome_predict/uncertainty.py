"""
Uncertainty quantification and the reject option (spec Section 5).

For a medical-device context, a bare class label is never enough. This module
turns the ensemble's per-member probabilities into:

* a mean probability per class,
* a disagreement (std) across models — a pLDDT-style per-prediction reliability,
* confidence summaries (top probability, margin, normalized entropy),
* prediction intervals, and
* a reject option that returns ``INDETERMINATE`` when confidence falls below a
  configurable threshold (e.g. < 0.90).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

INDETERMINATE = "INDETERMINATE"


def aggregate_members(member_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse ``(n_members, n_samples, n_classes)`` to mean and std arrays."""
    member_proba = np.asarray(member_proba, dtype=float)
    return member_proba.mean(axis=0), member_proba.std(axis=0)


def confidence_scores(mean_proba: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-sample confidence summaries from mean class probabilities."""
    p = np.asarray(mean_proba, dtype=float)
    top = p.max(axis=1)
    if p.shape[1] >= 2:
        ordered = np.sort(p, axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
    else:
        margin = top.copy()
    eps = 1e-12
    entropy = -(p * np.log(p + eps)).sum(axis=1)
    denom = np.log(p.shape[1]) if p.shape[1] > 1 else 1.0
    return {
        "max_proba": top,
        "margin": margin,
        "normalized_entropy": entropy / denom,
    }


def reject_mask(mean_proba: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Boolean mask: True where the top probability is below ``threshold``."""
    return np.asarray(mean_proba, dtype=float).max(axis=1) < threshold


def apply_reject_option(
    mean_proba: np.ndarray,
    classes: Sequence,
    threshold: float = 0.9,
    indeterminate_label: str = INDETERMINATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(labels, rejected_mask)`` with low-confidence calls abstained.

    Samples whose maximum class probability is below ``threshold`` are labeled
    ``indeterminate_label`` rather than forced into a class — the key safety
    feature that lets the system trade coverage for higher selective accuracy.
    """
    mean_proba = np.asarray(mean_proba, dtype=float)
    classes_arr = np.asarray(classes, dtype=object)
    labels = classes_arr[np.argmax(mean_proba, axis=1)].astype(object)
    rejected = reject_mask(mean_proba, threshold)
    labels[rejected] = indeterminate_label
    return labels, rejected


def prediction_interval(
    member_proba: np.ndarray, z: float = 1.96
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate ``(lower, mean, upper)`` probability bands from member spread.

    The interval width reflects how much the ensemble members disagree, giving
    each probability an explicit uncertainty envelope (clipped to ``[0, 1]``).
    """
    mean, std = aggregate_members(member_proba)
    lower = np.clip(mean - z * std, 0.0, 1.0)
    upper = np.clip(mean + z * std, 0.0, 1.0)
    return lower, mean, upper
