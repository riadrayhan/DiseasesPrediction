"""
Validation utilities (spec Section 5).

* ``cross_validate_classifier`` — stratified k-fold out-of-fold evaluation
  (accuracy, balanced accuracy, macro-F1, ROC-AUC).
* ``selective_prediction_metrics`` — accuracy vs. coverage under the reject
  option, demonstrating the accuracy/coverage trade-off.
* ``external_validation`` — fit on a training cohort, evaluate on a wholly
  independent cohort (the gold standard for generalizability claims).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _auc(y, proba, classes) -> float:
    try:
        if len(classes) == 2:
            return float(roc_auc_score((y == classes[1]).astype(int), proba[:, 1]))
        return float(roc_auc_score(y, proba, multi_class="ovr"))
    except Exception:
        return float("nan")


def _classification_metrics(y, proba, classes) -> Dict[str, float]:
    pred = classes[np.argmax(proba, axis=1)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "roc_auc": _auc(y, proba, classes),
    }


def cross_validate_classifier(
    clf,
    X,
    y,
    cv: int = 5,
    random_state: int = 0,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Stratified k-fold cross-validation returning ``(metrics, oof_proba)``."""
    y = np.asarray(y)
    classes = np.unique(y)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_proba = cross_val_predict(
        clone(clf), X, y, cv=splitter, method="predict_proba"
    )
    return _classification_metrics(y, oof_proba, classes), oof_proba


def selective_prediction_metrics(
    y_true,
    proba,
    classes,
    threshold: float = 0.9,
) -> Dict[str, float]:
    """Accuracy and coverage when abstaining below a confidence ``threshold``."""
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    confidence = proba.max(axis=1)
    pred = classes[np.argmax(proba, axis=1)]
    keep = confidence >= threshold
    coverage = float(keep.mean())
    selective_accuracy = (
        float(accuracy_score(y_true[keep], pred[keep])) if keep.any() else float("nan")
    )
    return {
        "threshold": float(threshold),
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "n_retained": int(keep.sum()),
        "n_total": int(len(y_true)),
    }


def external_validation(clf, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """Fit on a training cohort, evaluate on an independent test cohort."""
    fitted = clone(clf).fit(X_train, y_train)
    proba = fitted.predict_proba(X_test)
    classes = getattr(fitted, "classes_", np.unique(np.asarray(y_test)))
    return _classification_metrics(np.asarray(y_test), proba, np.asarray(classes))
