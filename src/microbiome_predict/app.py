"""
Optional Streamlit web frontend (spec Section 6).

Run with::

    pip install 'microbiome-predict[app]'
    streamlit run src/microbiome_predict/app.py

It lets a user upload a trained model bundle and an abundance matrix, runs the
prediction pipeline (probabilities, reject option, wellness index), shows the
results table, and offers the clinical report as an HTML or PDF download. It is
intentionally a thin UI over the same library functions the CLI uses.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def main() -> None:
    import streamlit as st

    from microbiome_predict import data, ingest, uncertainty
    from microbiome_predict.bundle import TrainedBundle
    from microbiome_predict.report import (
        ReportData,
        SamplePrediction,
        render_html_report,
    )

    st.set_page_config(page_title="Microbiome Disease Prediction", layout="wide")
    st.title("Microbiome Disease-Prediction")
    st.caption(
        "Decision-support software. Predictions are probabilistic inferences, "
        "not a standalone diagnosis."
    )

    with st.expander("Supported file types & how it works", expanded=False):
        st.markdown(
            "- **FASTA** (`.fna`, `.fasta`, `.fa`) / **FASTQ** (`.fastq`, `.fq`) — raw "
            "reads or contigs. The app computes a deterministic **QC summary** "
            "(read count, lengths, GC%, N50). If the headers carry taxonomy (NCBI "
            "binomial names or `s__Genus_species`), a species **profile** is derived "
            "from them. *Raw unannotated reads must be classified by the "
            "Kraken2/MetaPhlAn pipeline — the app never invents taxonomy.*\n"
            "- **Abundance matrix** (`.csv`) — samples x species, used directly.\n"
            "- **Taxonomic report** (`.tsv`, `.txt`) — MetaPhlAn/Kraken-style, or a "
            "2-column `species<TAB>abundance` table.\n"
            "- **Model bundle** (`.joblib`) — a model from `microbiome-predict train`.\n"
            "- `.gz`-compressed versions of any of the above are accepted.\n\n"
            "**To get predictions:** upload a model bundle **plus** at least one "
            "abundance/profile source."
        )

    uploads = st.file_uploader(
        "Upload files (FASTA/FASTQ, CSV/TSV, taxonomic report, or model .joblib)",
        type=["fna", "fasta", "fa", "ffn", "fastq", "fq", "csv", "tsv", "txt",
              "joblib", "pkl", "gz"],
        accept_multiple_files=True,
    )
    threshold = st.slider("Reject-option confidence threshold", 0.5, 0.99, 0.90, 0.01)

    if not uploads:
        st.info(
            "Upload at least one file to begin. A model bundle (.joblib) plus an "
            "abundance/profile source (CSV, taxonomic report, or annotated FASTA) "
            "is needed for predictions."
        )
        return

    bundle = None
    abundance_frames = []
    qc_frames = []
    for up in uploads:
        name = up.name
        raw = up.getvalue()
        if name.lower().endswith((".joblib", ".pkl")):
            try:
                bundle = TrainedBundle.load(io.BytesIO(raw))
                st.success(
                    f"Loaded model: **{name}** — {len(bundle.feature_names)} features; "
                    f"classes: {', '.join(map(str, bundle.classifier.classes_))}"
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load model bundle {name}: {exc}")
            continue
        result = ingest.ingest_bytes(name, raw)
        st.write(f"**{name}** &rarr; `{result.kind}`. {result.message}")
        if result.qc is not None:
            qc_frames.append(result.qc)
        if result.abundance is not None:
            abundance_frames.append(result.abundance)

    if qc_frames:
        st.subheader("Sequence QC summary")
        st.caption("Deterministic, sequencing-derived quality metrics.")
        st.dataframe(pd.concat(qc_frames, ignore_index=True), width="stretch")

    if not abundance_frames:
        st.info(
            "No species profile could be derived from the uploaded files yet. "
            "Upload an abundance matrix, a taxonomic report, or a taxonomy-annotated "
            "FASTA — or run raw reads through the Kraken2/MetaPhlAn pipeline first."
        )
        return

    matrix = pd.concat(abundance_frames, axis=0).fillna(0.0)
    matrix = matrix.groupby(level=0).sum()
    st.subheader("Derived feature matrix")
    st.caption(f"{matrix.shape[0]} sample(s) x {matrix.shape[1]} features "
               "(showing first 30 columns)")
    st.dataframe(matrix.iloc[:, :30], width="stretch")

    if bundle is None:
        st.warning("Add a trained model bundle (.joblib) above to run predictions.")
        return

    X = data.align_features(matrix, bundle.feature_names)
    overlap = int(matrix.columns.isin(bundle.feature_names).sum())
    st.caption(
        f"Feature alignment: {overlap}/{len(bundle.feature_names)} model features "
        "were present in the uploaded data."
    )
    if overlap == 0:
        st.warning(
            "None of the model's features were found in the uploaded data — this "
            "model was trained on a different taxonomy/feature set, so the prediction "
            "below is uninformative. Use a model trained on the same feature space as "
            "your input."
        )

    members = bundle.classifier.member_proba(X)
    lower, mean, upper = uncertainty.prediction_interval(members)
    classes = [str(c) for c in bundle.classifier.classes_]
    labels, rejected = uncertainty.apply_reject_option(
        mean, bundle.classifier.classes_, threshold=threshold
    )
    confidence = mean.max(axis=1)

    wellness = (
        bundle.wellness.score_samples(X) if bundle.wellness is not None else None
    )

    table = pd.DataFrame(
        {
            "sample": X.index,
            "prediction": labels,
            "confidence": confidence,
            "indeterminate": rejected,
        }
    )
    for j, cls in enumerate(classes):
        table[f"prob_{cls}"] = mean[:, j]
    if wellness is not None:
        table["wellness_index"] = wellness

    st.subheader("Predictions")
    st.dataframe(table, width="stretch")
    st.metric("Indeterminate (rejected)", int(np.sum(rejected)), f"of {len(X)}")

    samples = []
    for i, sample in enumerate(X.index):
        samples.append(
            SamplePrediction(
                sample=str(sample),
                predicted_label=str(labels[i]),
                probabilities={classes[j]: float(mean[i, j]) for j in range(len(classes))},
                confidence=float(confidence[i]),
                proba_interval={
                    classes[j]: (float(lower[i, j]), float(upper[i, j]))
                    for j in range(len(classes))
                },
                indeterminate=bool(rejected[i]),
                wellness_index=float(wellness[i]) if wellness is not None else None,
            )
        )
    report = ReportData(samples=samples, reject_threshold=threshold)

    st.download_button(
        "Download HTML report",
        data=render_html_report(report),
        file_name="report.html",
        mime="text/html",
    )

    try:
        from xhtml2pdf import pisa

        pdf_buf = io.BytesIO()
        pisa.CreatePDF(render_html_report(report, for_pdf=True), dest=pdf_buf)
        st.download_button(
            "Download PDF report",
            data=pdf_buf.getvalue(),
            file_name="report.pdf",
            mime="application/pdf",
        )
    except Exception:
        st.caption("Install the [pdf] extra for PDF export.")


if __name__ == "__main__":
    main()
