import numpy as np
import pytest

from microbiome_predict.models.deepsurv import _concordance_index


def test_concordance_index_perfect_and_worst():
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([1, 1, 1, 1])
    # Risk perfectly ordered with shorter survival -> C-index 1.0
    assert _concordance_index(durations, np.array([4, 3, 2, 1.0]), events) == 1.0
    # Risk reversed -> 0.0
    assert _concordance_index(durations, np.array([1, 2, 3, 4.0]), events) == 0.0


def test_deepsurv_trains_if_torch_present(synthetic_survival):
    pytest.importorskip("torch")
    from microbiome_predict.models.deepsurv import DeepSurvModel

    X, durations, events = synthetic_survival
    model = DeepSurvModel(hidden_layers=(16,), epochs=80).fit(X, durations, events)
    assert model.predict_risk(X).shape == (len(X),)
    assert model.concordance(X, durations, events) > 0.55
