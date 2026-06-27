# Intended Use & SaMD Classification (Starting Template)

> **Status:** Draft template for the development team / regulatory consultant.
> It is a structured starting point, **not** a cleared regulatory submission.
> Formal classification must be confirmed with qualified regulatory counsel.

## 1. Product

**Name:** microbiome-predict
**Description:** Software that analyzes microbial relative-abundance (and optional
functional/clinical) features derived from sequencing to produce probabilistic,
interpretable estimates of current disease status, disease risk, and
time-to-event (future) risk.

## 2. Intended Use Statement

microbiome-predict is intended to be used **as decision support** by qualified
healthcare professionals and researchers. It outputs probabilities, confidence
intervals, an explicit "indeterminate" (reject) result for low-confidence cases,
and the microbial features driving each result. It is **not** intended to be a
sole basis for diagnosis or treatment decisions.

## 3. Intended Users
- Clinical researchers and laboratory scientists.
- Healthcare professionals interpreting results alongside other clinical data.

## 4. Intended Patient Population
- Defined per validated indication and cohort (e.g., adults screened for
  colorectal cancer, IBD, or liver cirrhosis). Each indication requires its own
  validated model and performance claims.

## 5. SaMD Risk Categorization (IMDRF framework)
The IMDRF SaMD risk category is a function of (a) the significance of the
information to the healthcare decision and (b) the state of the healthcare
situation.

| Significance of information | Healthcare situation | Indicative SaMD category |
|---|---|---|
| Drives clinical management | Serious | III (higher risk) |
| Informs clinical management | Serious | II |
| Informs clinical management | Non-serious | I (lower risk) |

This product targets the **"informs clinical management"** row (decision support,
not autonomous diagnosis). The reject option and uncertainty reporting are
explicit risk-control measures keeping it out of the "treat or diagnose"
(autonomous) category.

## 6. Limitations (must be surfaced to users)
- Predictions are probabilistic inferences from one biological signal among many.
- Performance is indication- and population-specific; do not extrapolate beyond
  validated cohorts.
- Only the sequencing-derived taxonomic composition is deterministic; all disease
  predictions carry the reported uncertainty.

## 7. Applicable Standards (to be satisfied for clinical deployment)
- IEC 62304 — medical device software life-cycle (see `iec62304_software_lifecycle.md`).
- ISO 14971 — risk management (see `risk_analysis_iso14971.md`).
- IEC 62366-1 — usability engineering.
- ISO 13485 — quality management system.
- Regional regulation (e.g., FDA SaMD guidance, EU IVDR) as applicable.
