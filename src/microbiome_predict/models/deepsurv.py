"""
Model 4 (alternative) — DeepSurv neural survival model.

The spec names DeepSurv as an option alongside Cox PH for prognostic modeling.
This is a faithful, compact DeepSurv: a multilayer perceptron that outputs a
log-risk score trained to minimize the Cox negative log partial likelihood
(Breslow approximation).

PyTorch is an *optional* dependency. The class always imports; ``fit`` raises a
clear, actionable error if torch is absent. Install with::

    pip install torch        # CPU build is sufficient

For most tabular microbiome cohorts the penalized Cox model
(:class:`microbiome_predict.models.survival.PrognosticModel`) is a strong,
lighter-weight default — DeepSurv helps mainly with large samples and
non-linear risk surfaces.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..features import CLRTransformer

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover - environment without torch
    _HAS_TORCH = False


def torch_available() -> bool:
    return _HAS_TORCH


def _cox_ph_loss(log_risk, durations, events):
    """Cox negative log partial likelihood (Breslow tie handling)."""
    order = torch.argsort(durations, descending=True)
    log_risk = log_risk[order]
    events = events[order]
    # Cumulative log-sum-exp over the risk set (samples with longer-or-equal time).
    log_cumsum = torch.logcumsumexp(log_risk, dim=0)
    uncensored_ll = (log_risk - log_cumsum) * events
    n_events = torch.clamp(events.sum(), min=1.0)
    return -uncensored_ll.sum() / n_events


class DeepSurvModel:
    """MLP survival model trained on the Cox partial likelihood."""

    def __init__(
        self,
        hidden_layers: Sequence[int] = (32, 16),
        epochs: int = 300,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        use_clr: bool = True,
        random_state: int = 0,
    ):
        self.hidden_layers = tuple(hidden_layers)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.use_clr = use_clr
        self.random_state = random_state

    def _prepare(self, X, fit: bool = False) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if self.use_clr:
            arr = CLRTransformer().fit_transform(arr)
        if fit:
            self._mean = arr.mean(axis=0)
            self._std = arr.std(axis=0) + 1e-8
            self.feature_names_ = _feature_names(X)
        return (arr - self._mean) / self._std

    def _build_network(self, n_features: int):
        layers: List[nn.Module] = []
        prev = n_features
        for width in self.hidden_layers:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, durations: Sequence[float], events: Sequence[int]):
        if not _HAS_TORCH:
            raise ImportError(
                "DeepSurvModel requires PyTorch. Install with: pip install torch"
            )
        torch.manual_seed(self.random_state)
        features = self._prepare(X, fit=True)

        X_t = torch.tensor(features, dtype=torch.float32)
        dur_t = torch.tensor(np.asarray(durations, dtype=float), dtype=torch.float32)
        evt_t = torch.tensor(np.asarray(events, dtype=float), dtype=torch.float32)

        self.network_ = self._build_network(X_t.shape[1])
        optimizer = torch.optim.Adam(
            self.network_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.network_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            log_risk = self.network_(X_t).squeeze(-1)
            loss = _cox_ph_loss(log_risk, dur_t, evt_t)
            loss.backward()
            optimizer.step()
        self.final_loss_ = float(loss.item())
        return self

    def predict_risk(self, X) -> np.ndarray:
        """Relative risk score (exp of the network's log-risk output)."""
        if not _HAS_TORCH:
            raise ImportError("DeepSurvModel requires PyTorch.")
        self.network_.eval()
        features = self._prepare(X)
        with torch.no_grad():
            log_risk = self.network_(
                torch.tensor(features, dtype=torch.float32)
            ).squeeze(-1)
        return np.exp(log_risk.numpy())

    def concordance(self, X, durations, events) -> float:
        """Harrell's C-index of predicted risk vs observed survival."""
        risk = self.predict_risk(X)
        return _concordance_index(
            np.asarray(durations, dtype=float),
            risk,
            np.asarray(events, dtype=int),
        )


def _concordance_index(durations, risk, events) -> float:
    """Self-contained C-index (no lifelines dependency).

    Concordant if the higher-risk subject has the shorter survival time, over
    all comparable (one event, ordered times) pairs.
    """
    n = len(durations)
    concordant = 0.0
    permissible = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if durations[i] == durations[j]:
                continue
            shorter, longer = (i, j) if durations[i] < durations[j] else (j, i)
            if events[shorter] == 0:
                continue  # shorter time must be an observed event to be comparable
            permissible += 1
            if risk[shorter] > risk[longer]:
                concordant += 1
            elif risk[shorter] == risk[longer]:
                concordant += 0.5
    return concordant / permissible if permissible else float("nan")


def _feature_names(X) -> List[str]:
    if hasattr(X, "columns"):
        return list(X.columns)
    return [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
