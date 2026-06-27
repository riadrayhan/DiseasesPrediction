import pandas as pd

from microbiome_predict import cli


def _write_inputs(tmp_path, X, y):
    matrix_path = tmp_path / "matrix.csv"
    X.to_csv(matrix_path)
    meta = pd.DataFrame({"sample": X.index, "label": y.to_numpy()})
    meta_path = tmp_path / "meta.tsv"
    meta.to_csv(meta_path, sep="\t", index=False)
    return matrix_path, meta_path


def test_train_then_predict(tmp_path, synthetic_microbiome):
    X, y = synthetic_microbiome
    matrix_path, meta_path = _write_inputs(tmp_path, X, y)
    model_path = tmp_path / "model.joblib"

    rc = cli.main([
        "train", "--matrix", str(matrix_path), "--metadata", str(meta_path),
        "--label-col", "label", "--healthy-label", "0", "--cv", "0",
        "--out", str(model_path),
    ])
    assert rc == 0
    assert model_path.exists()

    # Predict on a small subset so per-sample occlusion explanations stay fast.
    small_path = tmp_path / "small.csv"
    X.iloc[:6].to_csv(small_path)
    report_path = tmp_path / "report.html"
    preds_path = tmp_path / "preds.tsv"

    rc = cli.main([
        "predict", "--model", str(model_path), "--matrix", str(small_path),
        "--report", str(report_path), "--predictions", str(preds_path),
        "--top-features", "8", "--max-explain-features", "20", "--threshold", "0.9",
        "--pdf",
    ])
    assert rc == 0
    assert report_path.exists()
    assert report_path.with_suffix(".pdf").exists()
    assert report_path.with_suffix(".pdf").read_bytes()[:4] == b"%PDF"
    assert "Current Disease Prediction" in report_path.read_text(encoding="utf-8")

    preds = pd.read_csv(preds_path, sep="\t")
    assert len(preds) == 6
    assert {"sample", "predicted_label", "confidence"}.issubset(preds.columns)
    assert "wellness_index" in preds.columns  # healthy label was provided


def test_crossval_cli(tmp_path, synthetic_microbiome, capsys):
    X, y = synthetic_microbiome
    matrix_path, meta_path = _write_inputs(tmp_path, X, y)
    rc = cli.main([
        "crossval", "--matrix", str(matrix_path), "--metadata", str(meta_path),
        "--label-col", "label", "--cv", "3", "--threshold", "0.9",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cross-validation" in out
    assert "selective_accuracy" in out


def test_rules_cli(tmp_path, synthetic_microbiome, capsys):
    X, y = synthetic_microbiome
    matrix_path, meta_path = _write_inputs(tmp_path, X, y)
    rules_path = tmp_path / "rules.txt"
    rc = cli.main([
        "rules", "--matrix", str(matrix_path), "--metadata", str(meta_path),
        "--label-col", "label", "--max-depth", "3", "--out", str(rules_path),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "host-status rules" in out
    assert rules_path.exists()
    assert "THEN" in rules_path.read_text(encoding="utf-8")
