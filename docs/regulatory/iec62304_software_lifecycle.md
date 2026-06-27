# IEC 62304 Software Life-Cycle Mapping (Starting Template)

> **Status:** Draft template mapping IEC 62304 activities to this repository.
> Establish the formal records, approvals, and configuration management under
> your QMS before any clinical use.

## Software Safety Classification (IEC 62304 §4.3)
Provisional class **B** (non-serious-injury possible if it fails), justified by
the decision-support intended use and the implemented risk controls (reject
option, mandatory clinician interpretation). Confirm per indication; a Class C
determination raises documentation rigor.

## Life-Cycle Activity Mapping

| IEC 62304 clause | Activity | Where it lives / status |
|---|---|---|
| §5.1 | Development planning | This document + `README.md` |
| §5.2 | Software requirements analysis | Project spec (Sections 1-7) → `docs/regulatory/intended_use_samd.md` |
| §5.3 | Software architectural design | Modular package: preprocessing (Snakemake) + `microbiome_predict` (data → features → models → uncertainty → interpret → report) |
| §5.4 | Detailed design | Module/class docstrings throughout `src/microbiome_predict/` |
| §5.5 | Unit implementation & verification | `src/` + `tests/` (automated suite) |
| §5.6 | Integration & integration testing | `tests/test_cli.py` (train→crossval→predict→rules end-to-end) |
| §5.7 | Software system testing | CLI/system tests + clinical validation plan (`validation_verification_plan.md`) |
| §5.8 | Software release | Versioned (`pyproject.toml`), containerized (`Dockerfile`, `Dockerfile.predict`) |
| §6 | Maintenance | Change control via version control + revalidation triggers |
| §7 | Risk management | `risk_analysis_iso14971.md` (ISO 14971) |
| §8 | Configuration management | Git; pinned dependencies in `pyproject.toml` / `requirements.txt`; conda `envs/` |
| §9 | Problem resolution | Issue tracker + post-market surveillance (see risk file) |

## SOUP (Software of Unknown Provenance) Inventory
Record version, intended use, and known anomalies for each third-party component.

| Component | Use | Pin |
|---|---|---|
| numpy, pandas | Data handling | `requirements.txt` / `pyproject.toml` |
| scikit-learn | Ensemble, metrics, CV | pinned |
| xgboost | Ensemble member | pinned |
| lifelines | Cox PH survival | pinned (extra) |
| torch | DeepSurv (optional) | pinned (extra) |
| xhtml2pdf | PDF export (optional) | pinned (extra) |
| Kraken2/Bracken/MetaPhlAn/Bowtie2/cutadapt/FastQC/MultiQC/HUMAnN3 | Preprocessing | conda `envs/*.yaml` |

## Traceability
Requirements (spec §3-§5) → modules (`models/*`, `uncertainty`, `interpret`,
`validation`, `report`) → tests (`tests/test_*`). Maintain a requirements-to-test
trace matrix as part of the design history file.
