"""Shared pytest fixtures: synthetic microbiome data (no real sequencing needed)."""

import numpy as np
import pandas as pd
import pytest


def _make_synthetic(seed: int = 42, n_per: int = 40, p: int = 60):
    """Generate a separable-but-noisy compositional microbiome dataset.

    Diseased samples (label 1) are enriched in the first 10 taxa and depleted in
    the next 10; the remaining taxa are noise. Rows are normalized to relative
    abundance and shuffled.
    """
    rng = np.random.default_rng(seed)

    def make(group: int) -> np.ndarray:
        base = rng.gamma(shape=0.5, scale=1.0, size=(n_per, p))
        if group == 1:
            base[:, :10] *= rng.uniform(2.5, 4.0, size=(n_per, 10))
            base[:, 10:20] *= rng.uniform(0.05, 0.3, size=(n_per, 10))
        return base

    arr = np.vstack([make(0), make(1)])
    arr = arr / arr.sum(axis=1, keepdims=True)

    n = 2 * n_per
    samples = [f"S{i:03d}" for i in range(n)]
    features = [f"Species_{j}" for j in range(p)]
    X = pd.DataFrame(arr, index=samples, columns=features)
    y = pd.Series([0] * n_per + [1] * n_per, index=samples, name="label")

    order = rng.permutation(n)
    return X.iloc[order], y.iloc[order]


@pytest.fixture
def synthetic_factory():
    """Return the generator so a test can build independent cohorts."""
    return _make_synthetic


@pytest.fixture
def synthetic_microbiome():
    return _make_synthetic(42)


@pytest.fixture(scope="session")
def fitted_ensemble():
    """A fitted ensemble shared across tests to avoid repeated training cost."""
    from microbiome_predict import EnsembleDiseaseClassifier

    X, y = _make_synthetic(7)
    clf = EnsembleDiseaseClassifier(min_prevalence=0.1, random_state=0).fit(X, y)
    return clf, X, y


@pytest.fixture
def synthetic_survival():
    """Microbiome features plus correlated (duration, event) outcomes."""
    X, _ = _make_synthetic(11)
    rng = np.random.default_rng(3)
    risk = X.iloc[:, :10].sum(axis=1).to_numpy()
    risk = (risk - risk.mean()) / (risk.std() + 1e-9)
    base = rng.exponential(scale=10.0, size=len(X))
    durations = np.clip(base * np.exp(-0.9 * risk), 0.1, None)
    events = (durations < np.median(durations)).astype(int)
    return (
        X,
        pd.Series(durations, index=X.index, name="duration"),
        pd.Series(events, index=X.index, name="event"),
    )
