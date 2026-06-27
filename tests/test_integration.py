import pandas as pd

from microbiome_predict import integration


def test_merge_omics_prefixes_and_joins():
    species = pd.DataFrame({"A": [1.0, 2.0], "B": [0.0, 1.0]}, index=["s1", "s2"])
    pathway = pd.DataFrame({"P1": [0.1, 0.2]}, index=["s1", "s2"])
    merged = integration.merge_omics({"species": species, "pathway": pathway})

    assert "species__A" in merged.columns
    assert "pathway__P1" in merged.columns
    assert list(merged.index) == ["s1", "s2"]
    assert integration.source_of("species__A") == "species"
    assert integration.source_of("pathway__P1") == "pathway"


def test_merge_omics_inner_join_drops_unshared_samples():
    a = pd.DataFrame({"A": [1.0, 2.0]}, index=["s1", "s2"])
    b = pd.DataFrame({"B": [3.0]}, index=["s1"])
    merged = integration.merge_omics({"a": a, "b": b}, how="inner")
    assert list(merged.index) == ["s1"]


def test_add_clinical_metadata_numeric_and_categorical():
    X = pd.DataFrame({"f": [0.5, 0.5]}, index=["s1", "s2"])
    meta = pd.DataFrame({"age": [40, 50], "sex": ["M", "F"]}, index=["s1", "s2"])
    out = integration.add_clinical_metadata(X, meta, ["age", "sex"])

    assert "clinical__age" in out.columns
    assert any(c.startswith("clinical__sex") for c in out.columns)
    assert out.shape[0] == 2
    assert "f" in out.columns  # original features preserved
