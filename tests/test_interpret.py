from microbiome_predict import interpret


def test_global_importance_finds_informative_taxa(fitted_ensemble):
    clf, X, y = fitted_ensemble
    importance = interpret.global_importance(clf, X, y, n_repeats=3)
    assert len(importance) == X.shape[1]
    # The signal lives in Species_0..19; at least one should rank in the top 15.
    informative = {f"Species_{i}" for i in range(20)}
    top15 = set(importance.head(15).index)
    assert informative & top15


def test_local_explanation_returns_top_k(fitted_ensemble):
    clf, X, y = fitted_ensemble
    expl = interpret.local_explanation(
        clf,
        X.iloc[0].to_numpy(),
        background_reference=X.mean(axis=0).to_numpy(),
        k=10,
        max_features_evaluated=20,
    )
    assert len(expl) <= 10
    assert expl.abs().is_monotonic_decreasing
