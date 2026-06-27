import numpy as np

from microbiome_predict import EnsembleDiseaseClassifier
from microbiome_predict.validation import (
    cross_validate_classifier,
    external_validation,
    selective_prediction_metrics,
)


def test_cross_validation_and_selective(synthetic_microbiome):
    X, y = synthetic_microbiome
    clf = EnsembleDiseaseClassifier(random_state=0)
    metrics, oof = cross_validate_classifier(clf, X, y, cv=3)
    assert oof.shape == (len(y), 2)
    assert metrics["roc_auc"] > 0.7

    selective = selective_prediction_metrics(y.to_numpy(), oof, np.unique(y),
                                              threshold=0.9)
    assert 0.0 <= selective["coverage"] <= 1.0
    assert selective["n_total"] == len(y)


def test_external_validation(synthetic_factory):
    X_tr, y_tr = synthetic_factory(1)
    X_te, y_te = synthetic_factory(2)
    metrics = external_validation(
        EnsembleDiseaseClassifier(random_state=0), X_tr, y_tr, X_te, y_te
    )
    assert metrics["roc_auc"] > 0.7
