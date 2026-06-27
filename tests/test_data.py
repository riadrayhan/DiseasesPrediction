import numpy as np
import pandas as pd

from microbiome_predict import data


def test_load_and_align(tmp_path):
    matrix = pd.DataFrame(
        {"Sp_A": [0.5, 0.2], "Sp_B": [0.5, 0.8]}, index=["s1", "s2"]
    )
    matrix.index.name = "sample"
    path = tmp_path / "m.csv"
    matrix.to_csv(path)

    loaded = data.load_abundance_matrix(path)
    assert list(loaded.index) == ["s1", "s2"]
    assert list(loaded.columns) == ["Sp_A", "Sp_B"]

    # Align onto a different feature universe: drop unknown, fill missing with 0.
    aligned = data.align_features(loaded, ["Sp_A", "Sp_C"])
    assert list(aligned.columns) == ["Sp_A", "Sp_C"]
    assert (aligned["Sp_C"] == 0.0).all()


def test_common_samples(tmp_path):
    matrix = pd.DataFrame({"Sp_A": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
    meta = pd.DataFrame({"label": [0, 1]}, index=["b", "c"])
    m, md = data.common_samples(matrix, meta)
    assert list(m.index) == ["b", "c"]
    assert list(md.index) == ["b", "c"]


def test_relative_abundance():
    matrix = pd.DataFrame({"a": [1.0, 0.0], "b": [3.0, 0.0]}, index=["x", "y"])
    rel = data.relative_abundance(matrix)
    np.testing.assert_allclose(rel.loc["x"].to_numpy(), [0.25, 0.75])
    # All-zero row stays all-zero (no division error).
    assert rel.loc["y"].sum() == 0.0
