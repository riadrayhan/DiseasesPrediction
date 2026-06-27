# Verification & Validation Plan (Starting Template)

> **Status:** Draft plan. Clinical performance claims require prospective,
> pre-registered validation on independent cohorts before any clinical use.

## 1. Verification (does the software work as designed?)

### 1.1 Automated test suite
- Location: `tests/` (run `python -m pytest`).
- Covers: feature transforms, all models (ensemble, wellness index, Cox PH,
  DeepSurv, interpretable rules), uncertainty/reject logic, interpretability,
  validation utilities, report + PDF rendering, multi-omics integration, the
  MetaPhlAn parser, and the full CLI (train → crossval → predict → rules).
- Exit criterion: 100% of tests pass on every change (CI gate).

### 1.2 Numerical / behavioral checks
- Probabilities sum to 1; CLR rows sum to 0; reject option abstains below
  threshold; prediction intervals bracket the mean; C-index sanity bounds.

## 2. Validation (does it meet clinical needs?)

### 2.1 Internal validation
- Stratified k-fold cross-validation (`validation.cross_validate_classifier`)
  reporting accuracy, balanced accuracy, macro-F1, ROC-AUC.
- Selective-prediction analysis (accuracy vs coverage at the reject threshold).

### 2.2 External validation (required)
- Evaluate the locked model on **independent cohorts not used in training**
  (`validation.external_validation`), per indication.
- Report discrimination (ROC-AUC), calibration (e.g., calibration curve /
  Brier score), and selective accuracy at the operating threshold.

### 2.3 Predefined acceptance criteria (example — set per indication)
| Metric | Target (illustrative) |
|---|---|
| External ROC-AUC | ≥ pre-registered threshold |
| Calibration (Brier) | ≤ pre-registered threshold |
| Selective accuracy @ ≥0.90 confidence | ≥ pre-registered threshold |
| Coverage @ operating threshold | reported (not optimized post-hoc) |

## 3. Data Management
- Document cohort provenance, inclusion/exclusion, label definitions, and
  preprocessing parameters (`config/config.yaml`).
- Freeze train/test splits; prevent leakage (prevalence filter and all transforms
  are fit on training folds only, enforced via the sklearn `Pipeline`).

## 4. Model Change Control
- Any change to features, model, or thresholds triggers re-verification (test
  suite) and re-validation (external cohort) before release.
- Record model version, training data hash, and metrics with each `TrainedBundle`.

## 5. Human Factors / Usability (IEC 62366-1)
- Validate that clinicians correctly interpret probabilities, intervals,
  "indeterminate" results, and the limitations disclaimer in the report.

## 6. Traceability
Maintain a matrix linking each requirement (spec §3-§5) to its verification test
and validation evidence as part of the design history file.
