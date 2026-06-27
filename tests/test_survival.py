import numpy as np
import pytest

pytest.importorskip("lifelines")

from microbiome_predict.models.survival import PrognosticModel  # noqa: E402


def test_prognostic_fit_and_concordance(synthetic_survival):
    X, durations, events = synthetic_survival
    model = PrognosticModel(top_k=15, penalizer=0.5).fit(X, durations, events)
    assert model.concordance_ > 0.6

    risk = model.predict_risk(X)
    assert risk.shape == (len(X),)
    # Independent concordance computation should also clear chance.
    assert model.concordance(X, durations, events) > 0.6


def test_prognostic_survival_function(synthetic_survival):
    X, durations, events = synthetic_survival
    model = PrognosticModel(top_k=10).fit(X, durations, events)
    surv = model.predict_survival_function(X.iloc[:3])
    # lifelines returns a DataFrame: index = timeline, columns = samples.
    assert surv.shape[1] == 3
    assert ((surv.to_numpy() >= 0) & (surv.to_numpy() <= 1)).all()
