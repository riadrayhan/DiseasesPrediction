"""
Optional Streamlit web frontend (spec Section 6).

Run with::

    pip install 'microbiome-predict[app]'
    streamlit run streamlit_app.py

Two modes:

* **Predict from sequences** — upload a raw ``.fna``/``.fastq``; the app reads the
  ATCG sequences, classifies the reads against a built-in reference panel
  (:mod:`microbiome_predict.classify`) to identify microbes, and predicts disease
  — no model file or database needed.
* **Advanced** — bring your own trained model bundle + abundance matrix / report.

It is a thin UI over the same library functions the CLI uses.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def _read_records(ingest, name, raw):
    """Parse a sequence upload into (records, ingest_result)."""
    text = ingest.decode_bytes(raw)
    kind = ingest.detect_kind(name, text[:256])
    records = (ingest.parse_fastq(text) if kind == ingest.KIND_FASTQ
               else ingest.parse_fasta(text))
    return records, ingest.ingest_bytes(name, raw)


def _predict_and_report(X, bundle, threshold, headline=True):
    """Run prediction on aligned features X; render results + report downloads."""
    import streamlit as st

    from microbiome_predict import uncertainty
    from microbiome_predict.report import (
        ReportData,
        SamplePrediction,
        render_html_report,
    )

    members = bundle.classifier.member_proba(X)
    lower, mean, upper = uncertainty.prediction_interval(members)
    classes = [str(c) for c in bundle.classifier.classes_]
    labels, rejected = uncertainty.apply_reject_option(
        mean, bundle.classifier.classes_, threshold=threshold
    )
    confidence = mean.max(axis=1)
    wellness = bundle.wellness.score_samples(X) if bundle.wellness is not None else None

    if headline:
        for i, sample in enumerate(X.index):
            if bool(rejected[i]):
                st.warning(f"**{sample} → INDETERMINATE** — top confidence "
                           f"{confidence[i]:.0%} is below the {threshold:.0%} threshold.")
            else:
                healthy = str(labels[i]).lower() in {"healthy", "control"}
                icon = "✅" if healthy else "⚠️"
                st.markdown(f"### {icon} {sample}: **{labels[i]}**  ·  "
                            f"{confidence[i]:.0%} confidence")

    table = pd.DataFrame({
        "sample": X.index,
        "prediction": labels,
        "confidence": confidence,
        "indeterminate": rejected,
    })
    for j, cls in enumerate(classes):
        table[f"prob_{cls}"] = mean[:, j]
    if wellness is not None:
        table["wellness_index"] = wellness
    st.dataframe(table, width="stretch")

    samples = []
    for i, sample in enumerate(X.index):
        samples.append(SamplePrediction(
            sample=str(sample),
            predicted_label=str(labels[i]),
            probabilities={classes[j]: float(mean[i, j]) for j in range(len(classes))},
            confidence=float(confidence[i]),
            proba_interval={classes[j]: (float(lower[i, j]), float(upper[i, j]))
                            for j in range(len(classes))},
            indeterminate=bool(rejected[i]),
            wellness_index=float(wellness[i]) if wellness is not None else None,
        ))
    report = ReportData(samples=samples, reject_threshold=threshold)

    col1, col2 = st.columns(2)
    col1.download_button("⬇️ Download HTML report",
                         data=render_html_report(report),
                         file_name="report.html", mime="text/html")
    try:
        from xhtml2pdf import pisa

        pdf_buf = io.BytesIO()
        pisa.CreatePDF(render_html_report(report, for_pdf=True), dest=pdf_buf)
        col2.download_button("⬇️ Download PDF report", data=pdf_buf.getvalue(),
                             file_name="report.pdf", mime="application/pdf")
    except Exception:
        col2.caption("Install the [pdf] extra for PDF export.")


def _sequence_tab():
    import streamlit as st

    from microbiome_predict import classify, data, ingest

    st.markdown(
        "Upload a **raw DNA/RNA sequence file** and get a disease prediction. The app:\n"
        "1. reads the **ATCG sequences** and runs quality control,\n"
        "2. **identifies which microbes** the reads come from (reference-based "
        "classification — the step that turns sequence into species),\n"
        "3. **predicts disease** from that microbe profile.\n\n"
        "No model file or database required — a built-in demo reference + model is used."
    )
    st.info(
        "The built-in reference is a compact **demonstration** panel of gut microbes "
        "(it identifies reads from those species). For real patient samples, run the "
        "Kraken2/MetaPhlAn pipeline in this repo against a full reference database, then "
        "use the Advanced tab.",
        icon="ℹ️",
    )

    @st.cache_resource(show_spinner="Building the built-in classifier and model…")
    def _engine():
        return classify.build_demo_classifier_and_model()

    file = st.file_uploader(
        "Sequence file (.fna / .fasta / .fastq, optionally .gz)",
        type=["fna", "fasta", "fa", "ffn", "fastq", "fq", "gz"], key="seq_file",
    )
    threshold = st.slider("Reporting confidence threshold", 0.5, 0.99, 0.60, 0.01,
                          key="seq_thr")

    if file is None:
        st.caption("No file? Create a demo patient sample in a terminal:  "
                   "`microbiome-predict classify --demo colorectal_cancer --out patient.fna`  "
                   "then upload patient.fna here.")
        return

    classifier, bundle = _engine()
    raw = file.getvalue()
    records, res = _read_records(ingest, file.name, raw)
    sample = file.name.split(".")[0]

    st.subheader("1 · Sequence quality control")
    if res.qc is not None:
        st.dataframe(res.qc, width="stretch")

    header_profile = res.abundance
    use_header = (header_profile is not None
                  and bool(header_profile.columns.isin(bundle.feature_names).any()))

    if use_header:
        st.subheader("2 · Microbes (from sequence-header taxonomy)")
        profile = header_profile
        st.caption("Taxonomy was present in the sequence headers — used directly.")
        st.bar_chart(profile.iloc[0].sort_values(ascending=False).head(12))
    else:
        st.subheader("2 · Identifying microbes (reference-based classification)")
        with st.spinner(f"Matching {len(records)} reads against the reference…"):
            profile, cstats = classifier.classify_profile(records, sample)
        if profile.empty or profile.to_numpy().sum() == 0:
            st.error(
                "None of the reads matched the built-in demo reference panel — the "
                "organisms in this file aren't in the demo set (e.g. a host/human genome, "
                "or real metagenomic reads). For real samples, use the Kraken2/MetaPhlAn "
                "pipeline with a full database (Advanced tab)."
            )
            return
        st.write(
            f"Identified **{cstats['n_classified']}/{cstats['n_reads']}** reads "
            f"({cstats['pct_classified']}%) across **{cstats['n_species_detected']}** "
            "species. Top microbes:"
        )
        st.bar_chart(profile.iloc[0].sort_values(ascending=False).head(12))

    st.subheader("3 · Disease prediction")
    X = data.align_features(profile, bundle.feature_names)
    _predict_and_report(X, bundle, threshold)


def _advanced_tab():
    import streamlit as st

    from microbiome_predict import data, ingest
    from microbiome_predict.bundle import TrainedBundle

    st.markdown(
        "Use **your own trained model** (`.joblib` from `microbiome-predict train`) plus "
        "one or more feature sources: an **abundance matrix** (`.csv`), a **taxonomic "
        "report** (`.tsv`/MetaPhlAn/Kraken), or a **taxonomy-annotated FASTA**. `.gz` ok."
    )
    uploads = st.file_uploader(
        "Files (model .joblib + matrix / report / FASTA)",
        type=["joblib", "pkl", "csv", "tsv", "txt", "fna", "fasta", "fa", "ffn",
              "fastq", "fq", "gz"],
        accept_multiple_files=True, key="adv_files",
    )
    threshold = st.slider("Reject-option confidence threshold", 0.5, 0.99, 0.90, 0.01,
                          key="adv_thr")
    if not uploads:
        st.caption("Upload a model bundle plus at least one abundance/profile source.")
        return

    bundle = None
    abundance_frames, qc_frames = [], []
    for up in uploads:
        name, raw = up.name, up.getvalue()
        if name.lower().endswith((".joblib", ".pkl")):
            try:
                bundle = TrainedBundle.load(io.BytesIO(raw))
                st.success(f"Loaded model **{name}** — {len(bundle.feature_names)} "
                           "features; classes: "
                           f"{', '.join(map(str, bundle.classifier.classes_))}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load {name}: {exc}")
            continue
        res = ingest.ingest_bytes(name, raw)
        st.write(f"**{name}** → `{res.kind}`. {res.message}")
        if res.qc is not None:
            qc_frames.append(res.qc)
        if res.abundance is not None:
            abundance_frames.append(res.abundance)

    if qc_frames:
        st.subheader("Sequence QC summary")
        st.dataframe(pd.concat(qc_frames, ignore_index=True), width="stretch")
    if not abundance_frames:
        st.info("No abundance/profile derived yet. Add a CSV matrix, a taxonomic "
                "report, or an annotated FASTA.")
        return

    matrix = pd.concat(abundance_frames, axis=0).fillna(0.0).groupby(level=0).sum()
    st.subheader("Derived feature matrix")
    st.caption(f"{matrix.shape[0]} sample(s) × {matrix.shape[1]} features (first 30)")
    st.dataframe(matrix.iloc[:, :30], width="stretch")

    if bundle is None:
        st.warning("Add a trained model bundle (.joblib) to run predictions.")
        return
    X = data.align_features(matrix, bundle.feature_names)
    overlap = int(matrix.columns.isin(bundle.feature_names).sum())
    st.caption(f"Feature alignment: {overlap}/{len(bundle.feature_names)} model "
               "features present in the uploaded data.")
    if overlap == 0:
        st.warning("None of the model's features were found — the model was trained on "
                   "a different feature set, so the prediction is uninformative.")
    st.subheader("Predictions")
    _predict_and_report(X, bundle, threshold)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Microbiome Disease Prediction", layout="wide")
    st.title("🧬 Microbiome Disease-Prediction")
    st.caption("Decision-support software. Predictions are probabilistic inferences "
               "from the microbiome, not a standalone diagnosis.")

    tab_seq, tab_adv = st.tabs(
        ["🧬 Predict from sequences (.fna / .fastq)",
         "⚙️ Advanced: your own model + data"]
    )
    with tab_seq:
        _sequence_tab()
    with tab_adv:
        _advanced_tab()


if __name__ == "__main__":
    main()
