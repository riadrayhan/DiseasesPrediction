# ==============================================================================
# Microbiome Preprocessing Pipeline — Container
# ==============================================================================
# Build:  docker build -t microbiome-pipeline .
# Run:    docker run -v $(pwd)/data:/data -v $(pwd)/resources:/resources \
#             microbiome-pipeline snakemake --cores 8 -p
#
# Note: this image installs the pipeline's *tools*. It does NOT bundle the
# Kraken2 database, host genome index, or HUMAnN3 databases — those are large
# (Kraken2 standard DB ~50-100GB, HUMAnN3 DBs ~20GB+) and must be mounted in
# or downloaded separately at runtime. See README.md "Reference data" section.
# ==============================================================================

FROM condaforge/mambaforge:24.3.0-0

LABEL maintainer="Riad Rayhan"
LABEL description="Microbiome preprocessing pipeline: QC -> trim -> host removal -> taxonomic profiling -> abundance matrix"

WORKDIR /pipeline

# Core workflow engine
RUN mamba install -y -c bioconda -c conda-forge \
    snakemake-minimal=8.* \
    pandas=2.* \
    && mamba clean -afy

# Per-step bioinformatics tools (kept in one image for simplicity here;
# the Snakefile is also written to use --use-conda with the per-rule
# envs/*.yaml files if you'd rather isolate them at runtime instead).
RUN mamba install -y -c bioconda -c conda-forge \
    fastqc=0.12.1 \
    multiqc=1.21 \
    cutadapt=4.6 \
    bowtie2=2.5.3 \
    samtools=1.19 \
    kraken2=2.1.3 \
    bracken=2.9 \
    && mamba clean -afy

COPY Snakefile config/ scripts/ envs/ /pipeline/

ENTRYPOINT ["snakemake"]
CMD ["--help"]
