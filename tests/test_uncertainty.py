import numpy as np

from microbiome_predict import uncertainty


def test_confidence_scores_ranges():
    mean = np.array([[0.95, 0.05], [0.5, 0.5]])
    scores = uncertainty.confidence_scores(mean)
    assert set(scores) == {"max_proba", "margin", "normalized_entropy"}
    np.testing.assert_allclose(scores["max_proba"], [0.95, 0.5])
    # Even split => maximal normalized entropy (~1), confident => ~0.
    assert scores["normalized_entropy"][1] > scores["normalized_entropy"][0]


def test_reject_option_marks_low_confidence():
    mean = np.array([[0.95, 0.05], [0.55, 0.45]])
    labels, rejected = uncertainty.apply_reject_option(mean, ["healthy", "disease"],
                                                       threshold=0.9)
    assert labels[0] == "healthy"
    assert labels[1] == "INDETERMINATE"
    assert rejected.tolist() == [False, True]


def test_prediction_interval_orders():
    # 3 members disagreeing on a 2-class problem.
    members = np.array(
        [
            [[0.8, 0.2]],
            [[0.6, 0.4]],
            [[0.7, 0.3]],
        ]
    )
    lower, mean, upper = uncertainty.prediction_interval(members)
    assert (lower <= mean).all()
    assert (mean <= upper).all()
    assert ((lower >= 0) & (upper <= 1)).all()
