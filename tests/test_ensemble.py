import numpy as np


def test_predict_proba_shape_and_normalization(fitted_ensemble):
    clf, X, y = fitted_ensemble
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y)), atol=1e-6)


def test_predict_labels_in_classes(fitted_ensemble):
    clf, X, y = fitted_ensemble
    preds = clf.predict(X)
    assert set(np.unique(preds)).issubset(set(clf.classes_))


def test_training_accuracy_reasonable(fitted_ensemble):
    clf, X, y = fitted_ensemble
    acc = (clf.predict(X) == y.to_numpy()).mean()
    assert acc > 0.85


def test_member_proba_shape(fitted_ensemble):
    clf, X, y = fitted_ensemble
    members = clf.member_proba(X)
    assert members.shape == (len(clf.member_names_), len(y), 2)
    # Soft-voting mean of members equals predict_proba.
    np.testing.assert_allclose(members.mean(axis=0), clf.predict_proba(X), atol=1e-6)
