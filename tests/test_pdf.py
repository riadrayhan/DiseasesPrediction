from microbiome_predict.pdf import pdf_available, write_pdf_report
from microbiome_predict.report import ReportData, SamplePrediction


def _sample():
    return SamplePrediction(
        sample="S001",
        predicted_label="colorectal_cancer",
        probabilities={"healthy": 0.2, "colorectal_cancer": 0.8},
        confidence=0.8,
        proba_interval={"healthy": (0.1, 0.3), "colorectal_cancer": (0.7, 0.9)},
        top_features=[("Fusobacterium_nucleatum", 0.12)],
    )


def test_pdf_available():
    assert pdf_available()


def test_write_pdf_report(tmp_path):
    out = write_pdf_report(ReportData(samples=[_sample()]), tmp_path / "r.pdf")
    assert out.exists()
    blob = out.read_bytes()
    assert blob[:4] == b"%PDF"
    assert len(blob) > 1000
