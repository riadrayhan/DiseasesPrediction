from microbiome_predict.models.interpretable import (
    InterpretableRuleClassifier,
    MMETHANEAdapter,
)


def test_rules_are_human_readable(synthetic_microbiome):
    X, y = synthetic_microbiome
    clf = InterpretableRuleClassifier(max_depth=3).fit(X, y)

    acc = (clf.predict(X) == y.to_numpy()).mean()
    assert acc > 0.75

    rules = clf.rules()
    assert len(rules) >= 1
    assert all("THEN" in r for r in rules)
    # Thresholds are rendered in relative-abundance percent.
    assert "%" in clf.rules_text()


def test_predict_proba_shape(synthetic_microbiome):
    X, y = synthetic_microbiome
    clf = InterpretableRuleClassifier(max_depth=2).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_mmethane_adapter_surfaces_rules(synthetic_microbiome):
    X, y = synthetic_microbiome
    base = InterpretableRuleClassifier(max_depth=2)
    adapter = MMETHANEAdapter(base).fit(X, y)
    assert adapter.predict_proba(X).shape[0] == len(X)
    assert len(adapter.predict(X)) == len(X)
    assert adapter.rules()  # non-empty, delegated to the wrapped model
