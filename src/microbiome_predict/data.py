"""
Data loading and alignment utilities.

These functions bridge the upstream preprocessing output (a sample x feature
abundance matrix) and the predictive models. They are deliberately free of any
heavy ML dependency so they can be imported and tested in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


def load_abundance_matrix(path: str | Path) -> pd.DataFrame:
    """Load a sample x feature abundance matrix (CSV).

    The first column is treated as the sample identifier index. Non-numeric or
    missing values are coerced to 0.0 so downstream models always receive a
    clean numeric matrix.
    """
    df = pd.read_csv(path, index_col=0)
    df.index.name = df.index.name or "sample"
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load sample metadata (CSV or TSV) indexed by a required ``sample`` column.

    Metadata typically holds the prediction label(s) and, for prognostic
    modeling, time-to-event columns (duration + event).
    """
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    meta = pd.read_csv(p, sep=sep)
    if "sample" not in meta.columns:
        raise ValueError(
            f"metadata file {p} must contain a 'sample' column "
            f"(found: {list(meta.columns)})"
        )
    return meta.set_index("sample")


def align_features(matrix: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Reindex an incoming matrix onto a trained model's feature set.

    Features the model has never seen are dropped; features the model expects
    but that are absent in this sample are filled with 0.0. This is essential
    for applying a trained model to a new sample whose taxonomic profile won't
    perfectly match the training feature universe.
    """
    aligned = matrix.reindex(columns=list(feature_names), fill_value=0.0)
    return aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def common_samples(
    matrix: pd.DataFrame, metadata: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to samples present in both, preserving matrix order."""
    meta_index = set(metadata.index)
    common = [s for s in matrix.index if s in meta_index]
    if not common:
        raise ValueError(
            "No overlapping samples between the abundance matrix and metadata. "
            "Check that sample identifiers match exactly."
        )
    return matrix.loc[common], metadata.loc[common]


def relative_abundance(matrix: pd.DataFrame) -> pd.DataFrame:
    """Row-normalize counts/abundances so each sample sums to 1.0."""
    arr = matrix.to_numpy(dtype=float)
    totals = arr.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return pd.DataFrame(arr / totals, index=matrix.index, columns=matrix.columns)
