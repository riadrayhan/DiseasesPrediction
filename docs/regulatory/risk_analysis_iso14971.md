# Risk Management File — ISO 14971 (Starting Template)

> **Status:** Draft template. Populate, review, and sign off under your QMS.
> Risk acceptability thresholds must be defined by the manufacturer.

## 1. Scope
Risk analysis for the microbiome-predict software, covering hazards arising from
software function, ML model behavior, and result interpretation.

## 2. Risk Management Process (summary)
1. Identify hazards & hazardous situations.
2. Estimate risk (severity × probability).
3. Apply risk controls (in priority order: inherent safe design → protective
   measures → information for safety).
4. Evaluate residual risk and overall benefit/risk.
5. Monitor in production (post-market surveillance).

## 3. Severity / Probability scales (example — calibrate to your QMS)
- **Severity:** 1 Negligible · 2 Minor · 3 Serious · 4 Critical · 5 Catastrophic
- **Probability:** 1 Improbable · 2 Remote · 3 Occasional · 4 Probable · 5 Frequent

## 4. Hazard Analysis (illustrative, non-exhaustive)

| ID | Hazard / hazardous situation | Potential harm | Sev | Prob | Risk controls implemented in software | Residual |
|----|------------------------------|----------------|-----|------|----------------------------------------|----------|
| H1 | **False negative** (misses disease) | Delayed diagnosis/treatment | 4 | 3 | Reject option (`INDETERMINATE` < threshold); calibrated probabilities; mandatory "decision-support only" disclaimer; external-cohort validation | Med |
| H2 | **False positive** | Unnecessary anxiety / workup | 3 | 3 | Probability + confidence interval reported, not a binary call; reject option; interpretability shows driving taxa for clinician review | Med |
| H3 | **Overconfident probability** (poor calibration) | Misplaced trust | 4 | 2 | Soft-voting ensemble; inter-model-disagreement intervals; selective-accuracy validation; calibration checks in `validation` | Low |
| H4 | **Distribution shift** (new cohort/lab) | Degraded accuracy | 3 | 3 | External validation required before deployment; feature alignment fills/drops unseen taxa; provenance via `integration.source_of` | Med |
| H5 | **Feature mismatch** train vs predict | Wrong/garbage output | 4 | 2 | `TrainedBundle` stores the feature universe; `align_features` reindexes deterministically | Low |
| H6 | **Silent input corruption** (bad matrix) | Wrong output | 4 | 2 | Numeric coercion + NaN→0; sample/feature shape logging; QC summary embedded in report | Med |
| H7 | **Use beyond intended population** | Invalid result | 4 | 2 | Intended-use statement; per-indication models; documented limitations in report disclaimer | Med |
| H8 | **Misinterpretation as diagnosis** | Inappropriate clinical action | 5 | 2 | Prominent non-removable disclaimer; "indeterminate" results; rationale (top features) shown | Med |

## 5. Risk Controls Traceability (to implementation)
- Reject option / uncertainty → `microbiome_predict/uncertainty.py`
- Probability calibration & validation → `microbiome_predict/validation.py`
- Interpretability (clinician review) → `microbiome_predict/interpret.py`, `models/interpretable.py`
- Feature-mismatch protection → `microbiome_predict/data.py`, `bundle.py`
- Disclaimer & confidence reporting → `microbiome_predict/report.py`

## 6. Overall Residual Risk / Benefit
To be evaluated after risk controls and clinical validation. Document the
benefit/risk conclusion and any risks disclosed to users.

## 7. Post-Market Surveillance
- Monitor input distributions and confidence/coverage over time.
- Track reported discordances with clinical ground truth.
- Trigger revalidation on drift or model update.
