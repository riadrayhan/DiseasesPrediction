"""
Command-line interface for the microbiome prediction layer.

Subcommands
-----------
* ``train``     — fit the ensemble (+ optional wellness index) and save a bundle.
* ``crossval``  — stratified k-fold evaluation with selective-prediction metrics.
* ``predict``   — score new samples and emit a clinical HTML report + TSV.

Examples
--------
    microbiome-predict train \
        --matrix results/matrix/species_abundance_matrix.csv \
        --metadata config/labels.tsv --label-col disease \
        --healthy-label healthy --out model.joblib

    microbiome-predict crossval \
        --matrix results/matrix/species_abundance_matrix.csv \
        --metadata config/labels.tsv --label-col disease --cv 5

    microbiome-predict predict \
        --model model.joblib \
        --matrix results/matrix/species_abundance_matrix.csv \
        --report report.html --predictions predictions.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from . import data, ingest, uncertainty
from .bundle import TrainedBundle
from .interpret import local_explanation
from .models.ensemble import EnsembleDiseaseClassifier
from .models.interpretable import InterpretableRuleClassifier
from .models.wellness_index import WellnessIndex
from .report import ReportData, SamplePrediction, write_report
from .validation import cross_validate_classifier, selective_prediction_metrics

_HEALTHY_ALIASES = {"0", "healthy", "control", "hc", "h", "normal"}


def _resolve_healthy_label(unique_labels, user_value: Optional[str]):
    """Map a user-provided (string) healthy label to the actual label value."""
    if user_value is not None:
        for lbl in unique_labels:
            if str(lbl) == str(user_value):
                return lbl
        raise SystemExit(
            f"--healthy-label {user_value!r} not found among labels "
            f"{[str(x) for x in unique_labels]}"
        )
    # Auto-infer from common aliases.
    for lbl in unique_labels:
        if str(lbl).strip().lower() in _HEALTHY_ALIASES:
            return lbl
    return None


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def _cmd_train(args: argparse.Namespace) -> int:
    matrix = data.load_abundance_matrix(args.matrix)
    metadata = data.load_metadata(args.metadata)
    matrix, metadata = data.common_samples(matrix, metadata)

    if args.label_col not in metadata.columns:
        raise SystemExit(
            f"label column {args.label_col!r} not in metadata "
            f"(have: {list(metadata.columns)})"
        )
    y = metadata[args.label_col].to_numpy()
    print(f"[train] {matrix.shape[0]} samples x {matrix.shape[1]} features; "
          f"classes={sorted(set(map(str, y)))}", file=sys.stderr)

    clf = EnsembleDiseaseClassifier(
        min_prevalence=args.min_prevalence, random_state=args.random_state
    )

    cv_metrics = {}
    if args.cv and args.cv > 1:
        cv_metrics, _ = cross_validate_classifier(clf, matrix, y, cv=args.cv,
                                                   random_state=args.random_state)
        print(f"[train] {args.cv}-fold CV: " +
              ", ".join(f"{k}={v:.3f}" for k, v in cv_metrics.items()),
              file=sys.stderr)

    clf.fit(matrix, y)

    wellness = None
    healthy_label = None
    if not args.no_wellness:
        healthy_label = _resolve_healthy_label(clf.classes_, args.healthy_label)
        if healthy_label is not None:
            wellness = WellnessIndex(healthy_label=healthy_label).fit(matrix, y)
            print(f"[train] wellness index fitted (healthy_label={healthy_label!r})",
                  file=sys.stderr)
        else:
            print("[train] wellness index skipped (no healthy label found; "
                  "pass --healthy-label)", file=sys.stderr)

    bundle = TrainedBundle(
        classifier=clf,
        feature_names=list(matrix.columns),
        label_col=args.label_col,
        background_reference=matrix.to_numpy(dtype=float).mean(axis=0),
        wellness=wellness,
        healthy_label=healthy_label,
        cv_metrics=cv_metrics,
    )
    out = bundle.save(args.out)
    print(f"[train] saved model bundle -> {out}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# crossval
# ---------------------------------------------------------------------------
def _cmd_crossval(args: argparse.Namespace) -> int:
    matrix = data.load_abundance_matrix(args.matrix)
    metadata = data.load_metadata(args.metadata)
    matrix, metadata = data.common_samples(matrix, metadata)
    y = metadata[args.label_col].to_numpy()

    clf = EnsembleDiseaseClassifier(
        min_prevalence=args.min_prevalence, random_state=args.random_state
    )
    metrics, oof = cross_validate_classifier(clf, matrix, y, cv=args.cv,
                                             random_state=args.random_state)
    classes = np.unique(y)
    selective = selective_prediction_metrics(y, oof, classes, threshold=args.threshold)

    print(f"=== {args.cv}-fold cross-validation ===")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"=== reject option @ confidence >= {args.threshold:.0%} ===")
    print(f"  coverage            : {selective['coverage']:.1%} "
          f"({selective['n_retained']}/{selective['n_total']})")
    print(f"  selective_accuracy  : {selective['selective_accuracy']:.4f}")
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
def _cmd_predict(args: argparse.Namespace) -> int:
    bundle = TrainedBundle.load(args.model)
    clf = bundle.classifier

    matrix = data.load_abundance_matrix(args.matrix)
    X = data.align_features(matrix, bundle.feature_names)

    members = clf.member_proba(X)
    lower, mean, upper = uncertainty.prediction_interval(members)
    classes = [str(c) for c in clf.classes_]
    labels, rejected = uncertainty.apply_reject_option(
        mean, clf.classes_, threshold=args.threshold
    )
    confidence = mean.max(axis=1)

    wellness_scores = None
    wellness_calls = None
    if bundle.wellness is not None:
        wellness_scores = bundle.wellness.score_samples(X)
        wellness_calls = bundle.wellness.predict(X)

    survival_model = None
    if args.survival_model:
        import joblib

        survival_model = joblib.load(args.survival_model)
    survival_risk = (
        survival_model.predict_risk(X) if survival_model is not None else None
    )

    qc_summary = None
    if args.qc_summary and Path(args.qc_summary).exists():
        qc_summary = pd.read_csv(args.qc_summary, sep="\t")

    samples: List[SamplePrediction] = []
    tsv_rows = []
    for i, sample in enumerate(X.index):
        prob_map = {classes[j]: float(mean[i, j]) for j in range(len(classes))}
        interval_map = {
            classes[j]: (float(lower[i, j]), float(upper[i, j]))
            for j in range(len(classes))
        }

        top_features = []
        if not args.no_explain:
            expl = local_explanation(
                clf,
                X.iloc[i].to_numpy(),
                background_reference=bundle.background_reference,
                k=args.top_features,
                max_features_evaluated=args.max_explain_features,
            )
            top_features = [(name, float(val)) for name, val in expl.items()]

        future_risk = None
        if survival_risk is not None:
            future_risk = {"event_relative_risk": float(survival_risk[i])}

        samples.append(
            SamplePrediction(
                sample=str(sample),
                predicted_label=str(labels[i]),
                probabilities=prob_map,
                confidence=float(confidence[i]),
                proba_interval=interval_map,
                indeterminate=bool(rejected[i]),
                wellness_index=(
                    float(wellness_scores[i]) if wellness_scores is not None else None
                ),
                wellness_call=(
                    str(wellness_calls[i]) if wellness_calls is not None else None
                ),
                top_features=top_features,
                future_risk=future_risk,
            )
        )

        row = {
            "sample": str(sample),
            "predicted_label": str(labels[i]),
            "indeterminate": bool(rejected[i]),
            "confidence": float(confidence[i]),
        }
        row.update({f"prob_{c}": float(mean[i, j]) for j, c in enumerate(classes)})
        if wellness_scores is not None:
            row["wellness_index"] = float(wellness_scores[i])
            row["wellness_call"] = str(wellness_calls[i])
        tsv_rows.append(row)

    model_info = {
        "Classes": ", ".join(classes),
        "Features (training)": str(len(bundle.feature_names)),
        "Ensemble members": ", ".join(clf.member_names_),
        "Wellness index": "yes" if bundle.wellness is not None else "no",
        "Reject threshold": f"{args.threshold:.0%}",
    }
    if bundle.cv_metrics:
        model_info["Cross-val ROC-AUC"] = f"{bundle.cv_metrics.get('roc_auc', float('nan')):.3f}"

    report = ReportData(
        samples=samples,
        qc_summary=qc_summary,
        model_info=model_info,
        reject_threshold=args.threshold,
    )
    out_report = write_report(report, args.report)
    print(f"[predict] wrote report -> {out_report}", file=sys.stderr)

    if args.pdf:
        from .pdf import write_pdf_report

        pdf_path = Path(args.report).with_suffix(".pdf")
        write_pdf_report(report, pdf_path)
        print(f"[predict] wrote PDF report -> {pdf_path}", file=sys.stderr)

    if args.predictions:
        pred_df = pd.DataFrame(tsv_rows)
        Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(args.predictions, sep="\t", index=False)
        print(f"[predict] wrote predictions -> {args.predictions}", file=sys.stderr)

    n_reject = int(np.sum(rejected))
    print(f"[predict] {len(samples)} samples scored; {n_reject} indeterminate "
          f"(< {args.threshold:.0%} confidence)", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# rules (Model 1 — interpretable host-status rules)
# ---------------------------------------------------------------------------
def _cmd_rules(args: argparse.Namespace) -> int:
    matrix = data.load_abundance_matrix(args.matrix)
    metadata = data.load_metadata(args.metadata)
    matrix, metadata = data.common_samples(matrix, metadata)
    y = metadata[args.label_col].to_numpy()

    model = InterpretableRuleClassifier(
        max_depth=args.max_depth, random_state=args.random_state
    ).fit(matrix, y)
    rules = model.rules()

    print("=== Interpretable host-status rules (Model 1, MMETHANE-style) ===")
    for rule in rules:
        print(f"  - {rule}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(rules), encoding="utf-8")
        print(f"[rules] wrote {len(rules)} rules -> {args.out}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# ingest (multi-format input -> QC + abundance matrix)
# ---------------------------------------------------------------------------
def _cmd_ingest(args: argparse.Namespace) -> int:
    result = ingest.ingest_path(args.input)
    if args.sample:
        if result.abundance is not None:
            result.abundance.index = [args.sample] * len(result.abundance)
            result.abundance.index.name = "sample"
        if result.qc is not None:
            result.qc["sample"] = args.sample

    print(f"=== Ingest: {result.filename}  (kind={result.kind}) ===")
    print(result.message)
    if result.qc is not None:
        print("\n-- QC summary --")
        print(result.qc.to_string(index=False))
    if result.abundance is not None:
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            result.abundance.to_csv(args.output)
            print(f"[ingest] wrote abundance matrix -> {args.output}", file=sys.stderr)
        else:
            print("\n-- Derived profile (top 15 by abundance) --")
            top = result.abundance.iloc[0].sort_values(ascending=False).head(15)
            print(top.to_string())
    elif args.output:
        print("[ingest] no abundance matrix could be derived; nothing written.",
              file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# classify (raw reads -> built-in reference -> profile -> disease prediction)
# ---------------------------------------------------------------------------
def _cmd_classify(args: argparse.Namespace) -> int:
    from . import classify

    classifier, bundle = classify.build_demo_classifier_and_model()

    if args.demo:
        if args.demo not in classify.DISEASE_DESIGN:
            raise SystemExit(
                f"--demo must be one of {list(classify.DISEASE_DESIGN)}"
            )
        design = classify.DISEASE_DESIGN[args.demo]
        abundances = {sp: 1.0 for sp in classifier.species}
        for sp in design.get("up", []):
            abundances[sp] = 8.0
        for sp in design.get("down", []):
            abundances[sp] = 0.2
        records = classify.simulate_reads(
            classifier.reference, abundances, n_reads=args.n_reads, seed=args.seed
        )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as fh:
                for header, seq in records:
                    fh.write(f">{header}\n{seq}\n")
            print(f"[classify] wrote {len(records)} demo reads -> {args.out}",
                  file=sys.stderr)
        sample = args.demo
    elif args.input:
        text = ingest.decode_bytes(Path(args.input).read_bytes())
        kind = ingest.detect_kind(args.input, text[:256])
        records = (ingest.parse_fastq(text) if kind == ingest.KIND_FASTQ
                   else ingest.parse_fasta(text))
        sample = Path(args.input).stem
    else:
        raise SystemExit("Provide --input <file> or --demo <condition>.")

    profile, stats = classifier.classify_profile(records, sample)
    print(f"=== Read classification ({sample}) ===")
    print(f"  reads={stats['n_reads']}  classified={stats['n_classified']} "
          f"({stats['pct_classified']}%)  species={stats['n_species_detected']}")
    if profile.empty:
        print("No reads matched the built-in demo reference panel.")
        return 1
    print("\n  Top microbes:")
    print(profile.iloc[0].sort_values(ascending=False).head(10).round(4).to_string())

    X = data.align_features(profile, bundle.feature_names)
    proba = bundle.classifier.predict_proba(X)[0]
    classes = [str(c) for c in bundle.classifier.classes_]
    pred = classes[int(np.argmax(proba))]
    print("\n=== Disease prediction ===")
    print(f"  Predicted: {pred}  ({max(proba):.0%} confidence)")
    for c, p in sorted(zip(classes, proba), key=lambda kv: -kv[1]):
        print(f"    {c:20s} {p:.3f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        profile.to_csv(args.output)
        print(f"[classify] wrote species profile -> {args.output}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microbiome-predict",
        description="Microbiome-based disease prediction: train / cross-validate / predict.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train the ensemble and save a model bundle.")
    p_train.add_argument("--matrix", required=True, help="Sample x feature abundance CSV.")
    p_train.add_argument("--metadata", required=True, help="Sample metadata CSV/TSV.")
    p_train.add_argument("--label-col", required=True, help="Metadata column with the class label.")
    p_train.add_argument("--out", required=True, help="Output model bundle (.joblib).")
    p_train.add_argument("--healthy-label", default=None,
                         help="Label value denoting healthy (for the wellness index).")
    p_train.add_argument("--min-prevalence", type=float, default=0.1)
    p_train.add_argument("--cv", type=int, default=5, help="CV folds during training (0 to skip).")
    p_train.add_argument("--no-wellness", action="store_true", help="Skip the wellness index.")
    p_train.add_argument("--random-state", type=int, default=0)
    p_train.set_defaults(func=_cmd_train)

    # crossval
    p_cv = sub.add_parser("crossval", help="Cross-validate with selective-prediction metrics.")
    p_cv.add_argument("--matrix", required=True)
    p_cv.add_argument("--metadata", required=True)
    p_cv.add_argument("--label-col", required=True)
    p_cv.add_argument("--cv", type=int, default=5)
    p_cv.add_argument("--threshold", type=float, default=0.9)
    p_cv.add_argument("--min-prevalence", type=float, default=0.1)
    p_cv.add_argument("--random-state", type=int, default=0)
    p_cv.set_defaults(func=_cmd_crossval)

    # predict
    p_pred = sub.add_parser("predict", help="Score samples and write a clinical report.")
    p_pred.add_argument("--model", required=True, help="Trained bundle (.joblib).")
    p_pred.add_argument("--matrix", required=True, help="Sample x feature abundance CSV.")
    p_pred.add_argument("--report", required=True, help="Output HTML report path.")
    p_pred.add_argument("--predictions", default=None, help="Optional output predictions TSV.")
    p_pred.add_argument("--threshold", type=float, default=0.9,
                        help="Reject option: abstain below this top-class confidence.")
    p_pred.add_argument("--qc-summary", default=None, help="Optional QC summary TSV to embed.")
    p_pred.add_argument("--survival-model", default=None,
                        help="Optional joblib PrognosticModel for future-risk forecasting.")
    p_pred.add_argument("--top-features", type=int, default=15)
    p_pred.add_argument("--max-explain-features", type=int, default=60)
    p_pred.add_argument("--no-explain", action="store_true",
                        help="Skip per-sample interpretability (faster).")
    p_pred.add_argument("--pdf", action="store_true",
                        help="Also write a PDF report (requires the [pdf] extra).")
    p_pred.set_defaults(func=_cmd_predict)

    # rules
    p_rules = sub.add_parser("rules", help="Print interpretable host-status rules (Model 1).")
    p_rules.add_argument("--matrix", required=True)
    p_rules.add_argument("--metadata", required=True)
    p_rules.add_argument("--label-col", required=True)
    p_rules.add_argument("--max-depth", type=int, default=3)
    p_rules.add_argument("--out", default=None, help="Optional path to save the rules text.")
    p_rules.add_argument("--random-state", type=int, default=0)
    p_rules.set_defaults(func=_cmd_rules)

    # ingest
    p_ingest = sub.add_parser("ingest",
                              help="Ingest a FASTA/FASTQ/report/CSV file -> QC + abundance matrix.")
    p_ingest.add_argument("--input", required=True,
                          help="Input file (.fna/.fasta/.fastq/.csv/.tsv, optionally .gz).")
    p_ingest.add_argument("--output", default=None,
                          help="Optional output abundance-matrix CSV (when a profile is derivable).")
    p_ingest.add_argument("--sample", default=None,
                          help="Override the sample name (default: input filename stem).")
    p_ingest.set_defaults(func=_cmd_ingest)

    # classify
    p_classify = sub.add_parser(
        "classify",
        help="Classify raw reads with the built-in reference and predict disease.")
    p_classify.add_argument("--input", default=None,
                            help="Input .fna/.fastq of raw reads to classify.")
    p_classify.add_argument("--demo", default=None,
                            help="Generate + classify a demo sample: healthy | colorectal_cancer | ibd")
    p_classify.add_argument("--out", default=None,
                            help="With --demo, write the generated raw reads to this .fna.")
    p_classify.add_argument("--output", default=None,
                            help="Write the derived species-profile matrix to this CSV.")
    p_classify.add_argument("--n-reads", type=int, default=600)
    p_classify.add_argument("--seed", type=int, default=1)
    p_classify.set_defaults(func=_cmd_classify)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
