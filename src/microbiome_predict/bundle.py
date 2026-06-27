"""
Trained-model bundle: serializes everything needed to score a new sample.

A single ``.joblib`` file holds the fitted ensemble, the (optional) wellness
index, the training feature universe, a background reference for local
explanations, and the metadata column mapping. Keeping these together prevents
train/predict feature-mismatch bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from .models.ensemble import EnsembleDiseaseClassifier
from .models.wellness_index import WellnessIndex


@dataclass
class TrainedBundle:
    classifier: EnsembleDiseaseClassifier
    feature_names: List[str]
    label_col: str
    background_reference: np.ndarray
    wellness: Optional[WellnessIndex] = None
    healthy_label: Optional[object] = None
    cv_metrics: Dict[str, float] = field(default_factory=dict)
    schema_version: int = 1

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out)
        return out

    @staticmethod
    def load(path: str | Path) -> "TrainedBundle":
        return joblib.load(path)
