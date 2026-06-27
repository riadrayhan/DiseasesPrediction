"""
Multi-omics feature integration (spec Section 5).

Predictive power improves when the taxonomic matrix is combined with other data
layers — functional pathways (HUMAnN3), metabolomics, and structured clinical
metadata. These helpers merge any number of ``sample x feature`` tables on the
sample index and append clinical covariates as model features, while preserving
provenance via a ``source__feature`` column-naming convention so downstream
interpretability can attribute importance back to each omics layer.
"""

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd


def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f"{prefix}{c}" for c in out.columns]
    return out


def merge_omics(
    tables: Dict[str, pd.DataFrame],
    how: str = "inner",
    prefix_sources: bool = True,
) -> pd.DataFrame:
    """Merge several ``sample x feature`` tables into one feature matrix.

    Parameters
    ----------
    tables:
        Mapping of source name -> DataFrame (e.g. ``{"species": ..., "pathway": ...}``).
    how:
        Join strategy across sample indices (``"inner"`` keeps only samples
        present in every layer; ``"outer"`` keeps all, filling gaps with 0).
    prefix_sources:
        Prefix each column with ``"<source>__"`` to retain layer provenance.
    """
    if not tables:
        raise ValueError("No tables provided to merge_omics().")

    frames = []
    for name, df in tables.items():
        cleaned = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if prefix_sources:
            cleaned = _prefix_columns(cleaned, f"{name}__")
        frames.append(cleaned)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.join(frame, how=how)
    return merged.fillna(0.0)


def add_clinical_metadata(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    columns: Sequence[str],
    one_hot: bool = True,
    prefix: str = "clinical__",
) -> pd.DataFrame:
    """Append selected clinical metadata columns to the feature matrix.

    Numeric columns are passed through (missing values imputed with the column
    mean); categorical columns are one-hot encoded by default.
    """
    aligned = metadata.reindex(matrix.index)
    selected = aligned[list(columns)]

    numeric = selected.select_dtypes(include="number")
    categorical = selected.select_dtypes(exclude="number")

    parts = []
    if not numeric.empty:
        filled = numeric.fillna(numeric.mean())
        parts.append(_prefix_columns(filled, prefix))
    if not categorical.empty:
        if one_hot:
            dummies = pd.get_dummies(
                categorical.astype("object"),
                prefix=[f"{prefix}{c}" for c in categorical.columns],
            )
            parts.append(dummies.astype(float))
        else:
            parts.append(_prefix_columns(categorical, prefix))

    if not parts:
        return matrix.copy()

    extra = pd.concat(parts, axis=1)
    return matrix.join(extra, how="left").fillna(0.0)


def source_of(feature_name: str) -> str:
    """Return the omics-layer source encoded in a ``source__feature`` name."""
    return feature_name.split("__", 1)[0] if "__" in feature_name else "unknown"
