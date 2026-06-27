import numpy as np

from microbiome_predict.models.wellness_index import (
    DISEASED,
    HEALTHY,
    INDETERMINATE,
    WellnessIndex,
)


def test_wellness_separates_classes(synthetic_microbiome):
    X, y = synthetic_microbiome
    wi = WellnessIndex(healthy_label=0).fit(X, y)
    scores = wi.score_samples(X)
    healthy_mean = scores[y.to_numpy() == 0].mean()
    diseased_mean = scores[y.to_numpy() == 1].mean()
    assert healthy_mean > diseased_mean


def test_wellness_predict_labels(synthetic_microbiome):
    X, y = synthetic_microbiome
    wi = WellnessIndex(healthy_label=0).fit(X, y)
    calls = wi.predict(X)
    assert set(np.unique(calls)).issubset({HEALTHY, DISEASED, INDETERMINATE})


def test_wellness_requires_both_classes(synthetic_microbiome):
    X, y = synthetic_microbiome
    import pytest

    with pytest.raises(ValueError):
        WellnessIndex(healthy_label=0).fit(X, np.zeros(len(y)))
