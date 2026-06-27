import numpy as np

from microbiome_predict.features import (
    CLRTransformer,
    PrevalenceFilter,
    RelativeAbundance,
)


def test_prevalence_filter_drops_rare_columns():
    # col 0 present in all rows, col 1 present in 1/4 rows, col 2 never present.
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 5.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    pf = PrevalenceFilter(min_prevalence=0.5).fit(X)
    assert pf.keep_.tolist() == [True, False, False]
    assert pf.transform(X).shape == (4, 1)


def test_prevalence_filter_never_empties():
    X = np.zeros((3, 4))
    pf = PrevalenceFilter(min_prevalence=0.9).fit(X)
    assert pf.keep_.all()


def test_clr_rows_sum_to_zero():
    rng = np.random.default_rng(0)
    X = rng.random((5, 8))
    out = CLRTransformer().fit_transform(X)
    np.testing.assert_allclose(out.sum(axis=1), np.zeros(5), atol=1e-9)


def test_relative_abundance_rows_sum_to_one():
    X = np.array([[1.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    out = RelativeAbundance().fit_transform(X)
    np.testing.assert_allclose(out[0], [0.25, 0.25, 0.5])
    assert out[1].sum() == 0.0
