from microbiome_predict.report import (
    ReportData,
    SamplePrediction,
    render_html_report,
    write_report,
)


def _sample():
    return SamplePrediction(
        sample="S001",
        predicted_label="colorectal_cancer",
        probabilities={"healthy": 0.18, "colorectal_cancer": 0.82},
        confidence=0.82,
        proba_interval={"healthy": (0.10, 0.26), "colorectal_cancer": (0.74, 0.90)},
        indeterminate=False,
        wellness_index=-0.42,
        wellness_call="DISEASED",
        top_features=[("Fusobacterium_nucleatum", 0.12), ("Bacteroides_fragilis", -0.05)],
    )


def test_render_contains_required_sections():
    report = ReportData(samples=[_sample()], reject_threshold=0.9)
    html = render_html_report(report)
    for needle in [
        "Current Disease Prediction",
        "Model Confidence",
        "Interpretability",
        "Future Disease Forecast",
        "S001",
        "Fusobacterium_nucleatum",
        "decision-support",  # disclaimer present
    ]:
        assert needle in html


def test_indeterminate_renders_reject_note():
    s = _sample()
    s.indeterminate = True
    s.confidence = 0.61
    html = render_html_report(ReportData(samples=[s]))
    assert "INDETERMINATE" in html
    assert "reject option" in html


def test_write_report(tmp_path):
    out = write_report(ReportData(samples=[_sample()]), tmp_path / "r.html")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
