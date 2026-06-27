"""
PDF export for clinical reports (spec Section 4 — "PDF or HTML").

Wraps the HTML report in a PDF using the pure-Python ``xhtml2pdf`` backend, so
no system libraries (wkhtmltopdf / GTK) are required. Install with::

    pip install 'microbiome-predict[pdf]'

HTML remains the primary, richer format; PDF is offered for archival / sign-off.
"""

from __future__ import annotations

from pathlib import Path

from .report import ReportData, render_html_report


def pdf_available() -> bool:
    try:
        import xhtml2pdf  # noqa: F401

        return True
    except Exception:
        return False


def html_to_pdf(html: str, path: str | Path) -> Path:
    """Render an HTML string to a PDF file."""
    try:
        from xhtml2pdf import pisa
    except Exception as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "PDF export requires xhtml2pdf. Install with: "
            "pip install 'microbiome-predict[pdf]'"
        ) from exc

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        status = pisa.CreatePDF(html, dest=fh)
    if status.err:
        raise RuntimeError(f"PDF generation failed with {status.err} error(s).")
    return out


def write_pdf_report(report: ReportData, path: str | Path) -> Path:
    """Render a :class:`ReportData` straight to a PDF file."""
    return html_to_pdf(render_html_report(report, for_pdf=True), path)
