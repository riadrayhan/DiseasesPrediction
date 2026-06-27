# Microbiome Preprocessing Pipeline

A reproducible, configurable Snakemake pipeline implementing Section 2 of the
microbiome disease-prediction project spec:

```
Raw reads → QC → Trimming → Host removal → Taxonomic profiling → Abundance matrix
                                                  ↘ (optional) Functional profiling
```

The output (`results/matrix/species_abundance_matrix.csv`) is the standardized
sample × feature matrix that feeds the downstream predictive models
(MMETHANE, GMWI2, ensemble classifiers, survival models).

## ⚠️ Important: FASTA (.fna) vs FASTQ — read this first

The original spec says input is `.fna` (FASTA). **Raw sequencing reads almost
always come as FASTQ (`.fastq`/`.fq`), not FASTA**, because FASTQ carries
per-base quality scores — and quality-based steps (FastQC quality plots,
quality trimming with cutadapt, Bowtie2's quality-aware alignment) all
*require* those scores. FASTA has no quality information at all.

This pipeline is built for **FASTQ input**, which is what you'll get from any
Illumina/Nanopore/PacBio run or public dataset (SRA, ENA, HMP, etc.). `.fna`
is the right format for things like:
- the **host reference genome** used in host removal (e.g. `GRCh38.fna`)
- assembled/contig output, if you ever add an assembly step

If you genuinely only have `.fna` reads with no quality scores (e.g. someone
handed you pre-assembled or already-cleaned data), the quality-based steps
(FastQC, quality trimming) aren't meaningful — you'd skip straight to
taxonomic classification. Flag this loudly to anyone before assuming
`.fna` input "just works" the same way.

## Pipeline steps

| Step | Tool | Rule(s) |
|---|---|---|
| 1. Raw QC | FastQC + MultiQC | `fastqc_raw`, `multiqc_raw` |
| 2. Trimming | cutadapt | `trim_reads` |
| 3. Host removal | Bowtie2 | `host_removal` |
| 4. Post-clean QC | FastQC + MultiQC | `fastqc_clean`, `multiqc_clean` |
| 5. Taxonomic profiling | Kraken2 → Bracken | `kraken2`, `bracken` |
| 6. Abundance matrix | custom Python | `build_abundance_matrix` |
| 7. QC summary table | custom Python | `qc_summary` |
| 8. (optional) Functional profiling | HUMAnN3 | `humann3`, `build_pathway_matrix` |

## Quick start

```bash
# 1. Install Snakemake (conda recommended)
conda install -c bioconda -c conda-forge snakemake-minimal pandas

# 2. Edit config/config.yaml:
#    - point bowtie2_index at a pre-built host genome index
#    - point kraken2_db at a built Kraken2 database
# 3. Edit config/samples.tsv with your sample names + FASTQ paths

# 4. Dry-run first (validates the DAG, no tools/data actually needed to check this works)
snakemake -n -p

# 5. Run for real, with per-rule conda environments auto-created
snakemake --cores 8 --use-conda -p
```

## Reference data you need to provide (not bundled — too large for a repo)

1. **Host genome Bowtie2 index** — build once:
   ```bash
   bowtie2-build GRCh38.fna resources/host_genome/host_index
   ```
2. **Kraken2 database** — either build your own or download a pre-built one
   (e.g. the standard or `k2_standard` DB from the Kraken2 docs). Point
   `taxonomic_profiling.kraken2_db` in `config.yaml` at it.
   - Also make sure a Bracken `*.kmer_distrib` file exists for your read
     length inside that DB directory (Bracken needs this; it ships with most
     pre-built Kraken2 DBs for common lengths like 100/150/250).
3. **(Optional) HUMAnN3 ChocoPhlAn + UniRef databases**, if you enable
   functional profiling — these are large (tens of GB).

## Output

```
results/
├── qc/
│   ├── fastqc_raw/<sample>/
│   ├── multiqc_raw/multiqc_report.html
│   ├── fastqc_clean/<sample>/
│   ├── multiqc_clean/multiqc_report.html
│   └── qc_summary.tsv          ← per-sample: raw/trimmed reads, % host, % classified
├── trimmed/<sample>_R{1,2}.trimmed.fastq.gz
├── host_removed/<sample>_R{1,2}.clean.fastq.gz
├── kraken2/<sample>.kreport
├── bracken/<sample>.bracken
└── matrix/
    └── species_abundance_matrix.csv   ← sample × species, ready for ML models
```

`species_abundance_matrix.csv` is what gets handed to Model 1–4 in the wider
project spec.

## Tested components (no real sequencing data required)

`scripts/build_abundance_matrix.py` and `scripts/build_qc_summary.py` are
pure Python/pandas and were verified against synthetic Bracken/cutadapt/
Bowtie2/Kraken2 output included as inline examples during development — you
can sanity-check them yourself the same way before plugging in real data.

The full Snakemake DAG (`snakemake -n -p`) was validated end-to-end against
a placeholder sample sheet and resolves all 8 pipeline stages without
errors — confirming the workflow structure itself is correct, independent of
whether the actual bioinformatics binaries (Kraken2, Bowtie2, etc.) are
installed yet.

## Why Snakemake (vs. Nextflow)

Both were named as options in the spec. Snakemake was chosen here because:
- Pure Python rule definitions (easier to extend with custom logic like the
  matrix-builder script, which you'll likely keep iterating on)
- Native `--use-conda` per-rule environment isolation (used above) without
  needing a separate container per process
- Simpler local dry-run/debug loop

If your team is more Nextflow-native (e.g. nf-core has mature, pre-built
metagenomics pipelines like `nf-core/mag` and `nf-core/taxprofiler` that
overlap significantly with what's built here), porting this rule logic to
Nextflow DSL2 processes would be a reasonably direct translation — worth
considering rather than reinventing, since `nf-core/taxprofiler` already
wraps Kraken2/Bracken/MetaPhlAn with established best practices.

## Part 2 — Predictive modeling layer (`microbiome_predict`)

The `species_abundance_matrix.csv` produced above feeds a Python package,
`microbiome_predict` (under [src/microbiome_predict/](src/microbiome_predict)),
that turns abundances into clinically interpretable predictions. It implements
Sections 3–7 of the spec.

| Spec | Component | Module |
|---|---|---|
| Model 1 — interpretable AI | `InterpretableRuleClassifier` + `MMETHANEAdapter` (human-readable IF/THEN rules) | [models/interpretable.py](src/microbiome_predict/models/interpretable.py) |
| Model 2 — health indexing | `WellnessIndex` (GMWI2-style) | [models/wellness_index.py](src/microbiome_predict/models/wellness_index.py) |
| Model 3 — ensemble ML | `EnsembleDiseaseClassifier` (RF + SVM + GBM + XGBoost soft-vote) | [models/ensemble.py](src/microbiome_predict/models/ensemble.py) |
| Model 4 — prognostic | `PrognosticModel` (Cox PH) · `DeepSurvModel` (neural) | [models/survival.py](src/microbiome_predict/models/survival.py) · [models/deepsurv.py](src/microbiome_predict/models/deepsurv.py) |
| §5 — multi-omics integration | `integration` (merge species + pathways + clinical metadata) | [integration.py](src/microbiome_predict/integration.py) |
| §5 — reject option, intervals | `uncertainty` | [uncertainty.py](src/microbiome_predict/uncertainty.py) |
| §4/6 — interpretability | `interpret` (permutation + occlusion/SHAP) | [interpret.py](src/microbiome_predict/interpret.py) |
| §5 — validation | `validation` (CV, selective, external) | [validation.py](src/microbiome_predict/validation.py) |
| §4 — clinical report | `report` (HTML) + `pdf` (PDF export) | [report.py](src/microbiome_predict/report.py) · [pdf.py](src/microbiome_predict/pdf.py) |

Features are handled **compositionally** (prevalence filter → centered-log-ratio
→ standardize) before modeling, which is the statistically correct treatment
for simplex-constrained microbiome data.

### Install

```bash
pip install -e .                      # core (numpy, pandas, scikit-learn, xgboost)
pip install -e '.[survival,interpret,pdf,dev]'   # + lifelines, shap, xhtml2pdf, pytest
pip install -e '.[deepsurv,app]'      # + torch (DeepSurv), streamlit (web UI)
```

### Use

```bash
# Train an ensemble (+ wellness index) and save a model bundle.
microbiome-predict train \
    --matrix results/matrix/species_abundance_matrix.csv \
    --metadata config/labels.tsv --label-col disease \
    --healthy-label healthy --out model.joblib

# Honest validation: k-fold CV + selective-accuracy under the reject option.
microbiome-predict crossval \
    --matrix results/matrix/species_abundance_matrix.csv \
    --metadata config/labels.tsv --label-col disease --cv 5 --threshold 0.9

# Score new samples -> clinical HTML (+ optional PDF) report + predictions TSV.
microbiome-predict predict \
    --model model.joblib \
    --matrix results/matrix/species_abundance_matrix.csv \
    --report report.html --predictions predictions.tsv \
    --qc-summary results/qc/qc_summary.tsv --threshold 0.9 --pdf

# Print interpretable host-status rules (Model 1, MMETHANE-style).
microbiome-predict rules \
    --matrix results/matrix/species_abundance_matrix.csv \
    --metadata config/labels.tsv --label-col disease
```

### Additional capabilities

- **Multi-omics integration** — `integration.merge_omics({"species": ..., "pathway": ...})`
  and `integration.add_clinical_metadata(...)` fuse taxonomic, functional and
  clinical features into one matrix with `source__feature` provenance.
- **MetaPhlAn profiler** — set `taxonomic_profiling.profiler: metaphlan` in
  [config/config.yaml](config/config.yaml) to use MetaPhlAn instead of
  Kraken2/Bracken (parsed by [scripts/build_metaphlan_matrix.py](scripts/build_metaphlan_matrix.py)).
- **FASTA (.fna) input** — set `input_format: fasta`; the MetaPhlAn path classifies
  FASTA directly (quality-based trimming is skipped, as FASTA has no qualities).
- **Web UI** — `streamlit run src/microbiome_predict/app.py` (needs the `[app]` extra).
- **Containers** — preprocessing image ([Dockerfile](Dockerfile)) and ML-layer
  image ([Dockerfile.predict](Dockerfile.predict)).
- **Regulatory templates** — SaMD intended-use, ISO 14971 risk file, IEC 62304
  mapping and a V&V plan under [docs/regulatory/](docs/regulatory).

### Deploy the web app

The app is deployment-ready — entry point [streamlit_app.py](streamlit_app.py),
a lean [requirements.txt](requirements.txt), and [.streamlit/config.toml](.streamlit/config.toml)
(large upload limit for big FASTA/FASTQ files):

- **Streamlit Community Cloud** (free, easiest): push to GitHub, then at
  [share.streamlit.io](https://share.streamlit.io) → *New app* → select this repo
  and set the main file to `streamlit_app.py`.
- **Render**: the included [render.yaml](render.yaml) is a one-click Blueprint.
- **Railway / Heroku**: uses the [Procfile](Procfile).
- **Docker** (Render / Railway / Fly.io / Cloud Run):
  `docker build -f Dockerfile.app -t microbiome-predict-app . && docker run -p 8501:8501 microbiome-predict-app`

`config/labels.tsv` is a sample sheet with a `sample` column plus your label
column (and, for prognostic modeling, `duration`/`event` columns).

### What the report contains

The HTML report ([report.py](src/microbiome_predict/report.py)) renders every
section the spec asks for: data-quality summary, current prediction with a
probability **interval**, wellness-index risk analysis, future-disease forecast
(when a `--survival-model` is supplied), per-prediction **confidence**, and the
top contributing taxa for **interpretability** — under a prominent medical
disclaimer.

### On accuracy and the "reject option"

As the spec itself states, guaranteeing 98–100% accuracy from microbiome data
alone is not biologically realistic. This layer is built to be **honest about
uncertainty** instead:

- every prediction is a probability with an explicit interval derived from
  inter-model disagreement (a pLDDT-style reliability signal);
- the **reject option** abstains (`INDETERMINATE`) below a confidence threshold
  (default 0.90), trading coverage for higher selective accuracy — exactly the
  GMWI2 high-confidence behavior;
- validation supports internal CV **and** external-cohort evaluation, the only
  credible basis for generalizability claims.

This software is decision-support, not a standalone diagnostic. For SaMD use it
would need the IEC 62304 / ISO 14971 process, documentation, and clinical
validation described in spec Section 7.

### Testing (no real sequencing data required)

```bash
python -m pytest      # 40 tests over synthetic compositional data
```

The suite (under [tests/](tests)) exercises every module — feature transforms,
all models (ensemble, wellness index, Cox PH, DeepSurv, interpretable rules),
uncertainty/reject logic, interpretability, validation, multi-omics integration,
HTML+PDF reporting, the MetaPhlAn parser, and the full train→predict→rules CLI —
against synthetic data, so it runs anywhere without Kraken2/Bowtie2 or real
cohorts.

## Not bundled (optional external adapters)

- **Model 1 — MMETHANE** and the **canonical published GMWI2** are external,
  separately-licensed projects. This repo ships working *equivalents* —
  `InterpretableRuleClassifier` (glass-box host-status rules) and a GMWI2-style
  `WellnessIndex` — plus `MMETHANEAdapter`, so a real MMETHANE or the published
  GMWI2 coefficients can be dropped in behind the same
  `fit`/`predict_proba`/`rules` interface.
- **DeepSurv** needs PyTorch (`pip install 'microbiome-predict[deepsurv]'`); the
  lighter Cox PH `PrognosticModel` is the default prognostic model.
- The regulatory documents under [docs/regulatory/](docs/regulatory) are
  **starting templates**, not a cleared submission — clinical performance claims
  require prospective external validation.
