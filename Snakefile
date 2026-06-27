"""
Microbiome Preprocessing Pipeline
==================================
QC -> Trimming -> Host removal -> Taxonomic profiling -> Abundance matrix
(-> optional functional profiling)

Usage
-----
    snakemake --cores 8 --use-conda -p

Dry run (recommended first, doesn't need any real data or tools installed):
    snakemake -n -p

See README.md for full setup instructions.
"""

import pandas as pd
import os

configfile: "config/config.yaml"

# ------------------------------------------------------------------------------
# Load sample sheet
# ------------------------------------------------------------------------------
samples_df = pd.read_csv(config["samples_tsv"], sep="\t", dtype=str).fillna("")
samples_df = samples_df.set_index("sample", drop=False)
SAMPLES = list(samples_df["sample"])

def is_paired(sample):
    return samples_df.loc[sample, "fq2"].strip() != ""

def fq1_of(sample):
    return samples_df.loc[sample, "fq1"]

def fq2_of(sample):
    return samples_df.loc[sample, "fq2"]

OUT = config["output_dir"]
LOG = config["log_dir"]

# Active taxonomic profiler ("kraken2" | "metaphlan") and input sequence format
# ("fastq" | "fasta"); see config.yaml.
PROFILER = config["taxonomic_profiling"].get("profiler", "kraken2")
INPUT_FORMAT = config.get("input_format", "fastq")

# ------------------------------------------------------------------------------
# Target rule
# ------------------------------------------------------------------------------
def all_targets():
    targets = [
        f"{OUT}/qc/multiqc_raw/multiqc_report.html",
        f"{OUT}/qc/multiqc_clean/multiqc_report.html",
        f"{OUT}/matrix/species_abundance_matrix.csv",
    ]
    # qc_summary parses Kraken2 reports, so it only applies to the Kraken2 path.
    if PROFILER == "kraken2":
        targets.append(f"{OUT}/qc/qc_summary.tsv")
    if config["functional_profiling"]["enabled"]:
        targets.append(f"{OUT}/matrix/pathway_abundance_matrix.csv")
    return targets

rule all:
    input:
        all_targets()

# ==============================================================================
# Step 1: FastQC on raw reads
# ==============================================================================
rule fastqc_raw:
    input:
        r1=lambda wc: fq1_of(wc.sample),
        r2=lambda wc: fq2_of(wc.sample) if is_paired(wc.sample) else fq1_of(wc.sample),
    output:
        directory(f"{OUT}/qc/fastqc_raw/{{sample}}")
    log:
        f"{LOG}/fastqc_raw/{{sample}}.log"
    threads: 2
    conda:
        "envs/qc.yaml"
    shell:
        """
        mkdir -p {output}
        fastqc -o {output} -t {threads} {input.r1} {input.r2} > {log} 2>&1
        """

rule multiqc_raw:
    input:
        expand(f"{OUT}/qc/fastqc_raw/{{sample}}", sample=SAMPLES)
    output:
        f"{OUT}/qc/multiqc_raw/multiqc_report.html"
    log:
        f"{LOG}/multiqc_raw.log"
    conda:
        "envs/qc.yaml"
    shell:
        """
        multiqc {OUT}/qc/fastqc_raw -o {OUT}/qc/multiqc_raw > {log} 2>&1
        """

# ==============================================================================
# Step 2: Trimming (cutadapt)
# ==============================================================================
rule trim_reads:
    input:
        r1=lambda wc: fq1_of(wc.sample),
        r2=lambda wc: fq2_of(wc.sample) if is_paired(wc.sample) else [],
    output:
        r1=f"{OUT}/trimmed/{{sample}}_R1.trimmed.fastq.gz",
        r2=f"{OUT}/trimmed/{{sample}}_R2.trimmed.fastq.gz",
        report=f"{OUT}/trimmed/{{sample}}.cutadapt.json",
    log:
        f"{LOG}/trim/{{sample}}.log"
    threads: config["trimming"]["threads"]
    conda:
        "envs/trimming.yaml"
    params:
        a=config["trimming"]["adapter_r1"],
        A=config["trimming"]["adapter_r2"],
        q=config["trimming"]["quality_cutoff"],
        m=config["trimming"]["min_length"],
    shell:
        """
        if [ -n "{input.r2}" ]; then
            cutadapt -j {threads} \
                -a {params.a} -A {params.A} \
                -q {params.q} -m {params.m} \
                --json {output.report} \
                -o {output.r1} -p {output.r2} \
                {input.r1} {input.r2} > {log} 2>&1
        else
            cutadapt -j {threads} \
                -a {params.a} \
                -q {params.q} -m {params.m} \
                --json {output.report} \
                -o {output.r1} \
                {input.r1} > {log} 2>&1
            touch {output.r2}
        fi
        """

# ==============================================================================
# Step 3: Host read removal (Bowtie2 -> keep unmapped reads)
# ==============================================================================
rule host_removal:
    input:
        r1=f"{OUT}/trimmed/{{sample}}_R1.trimmed.fastq.gz",
        r2=f"{OUT}/trimmed/{{sample}}_R2.trimmed.fastq.gz",
    output:
        r1=f"{OUT}/host_removed/{{sample}}_R1.clean.fastq.gz",
        r2=f"{OUT}/host_removed/{{sample}}_R2.clean.fastq.gz",
        stats=f"{OUT}/host_removed/{{sample}}.bowtie2.stats.txt",
    log:
        f"{LOG}/host_removal/{{sample}}.log"
    threads: config["host_removal"]["threads"]
    conda:
        "envs/host_removal.yaml"
    params:
        idx=config["host_removal"]["bowtie2_index"],
        prefix=f"{OUT}/host_removed/{{sample}}",
    shell:
        """
        bowtie2 -x {params.idx} \
            -1 {input.r1} -2 {input.r2} \
            -p {threads} \
            --un-conc-gz {params.prefix}_unmapped_%.fastq.gz \
            -S /dev/null \
            2> {output.stats} | tee -a {log}

        mv {params.prefix}_unmapped_1.fastq.gz {output.r1}
        mv {params.prefix}_unmapped_2.fastq.gz {output.r2}
        """

# ==============================================================================
# Step 4: FastQC + MultiQC on cleaned reads (post host-removal, post-trim)
# ==============================================================================
rule fastqc_clean:
    input:
        r1=f"{OUT}/host_removed/{{sample}}_R1.clean.fastq.gz",
        r2=f"{OUT}/host_removed/{{sample}}_R2.clean.fastq.gz",
    output:
        directory(f"{OUT}/qc/fastqc_clean/{{sample}}")
    log:
        f"{LOG}/fastqc_clean/{{sample}}.log"
    threads: 2
    conda:
        "envs/qc.yaml"
    shell:
        """
        mkdir -p {output}
        fastqc -o {output} -t {threads} {input.r1} {input.r2} > {log} 2>&1
        """

rule multiqc_clean:
    input:
        expand(f"{OUT}/qc/fastqc_clean/{{sample}}", sample=SAMPLES)
    output:
        f"{OUT}/qc/multiqc_clean/multiqc_report.html"
    log:
        f"{LOG}/multiqc_clean.log"
    conda:
        "envs/qc.yaml"
    shell:
        """
        multiqc {OUT}/qc/fastqc_clean -o {OUT}/qc/multiqc_clean > {log} 2>&1
        """

# ==============================================================================
# Step 5: Taxonomic profiling — Kraken2 then Bracken re-estimation
# ==============================================================================
rule kraken2:
    input:
        r1=f"{OUT}/host_removed/{{sample}}_R1.clean.fastq.gz",
        r2=f"{OUT}/host_removed/{{sample}}_R2.clean.fastq.gz",
    output:
        report=f"{OUT}/kraken2/{{sample}}.kreport",
        out=f"{OUT}/kraken2/{{sample}}.kraken",
    log:
        f"{LOG}/kraken2/{{sample}}.log"
    threads: config["taxonomic_profiling"]["threads"]
    conda:
        "envs/taxonomy.yaml"
    params:
        db=config["taxonomic_profiling"]["kraken2_db"],
        conf=config["taxonomic_profiling"]["confidence"],
    shell:
        """
        kraken2 --db {params.db} \
            --threads {threads} \
            --confidence {params.conf} \
            --paired {input.r1} {input.r2} \
            --report {output.report} \
            --output {output.out} > {log} 2>&1
        """

rule bracken:
    input:
        report=f"{OUT}/kraken2/{{sample}}.kreport",
    output:
        bracken=f"{OUT}/bracken/{{sample}}.bracken",
    log:
        f"{LOG}/bracken/{{sample}}.log"
    conda:
        "envs/taxonomy.yaml"
    params:
        db=config["taxonomic_profiling"]["kraken2_db"],
        read_len=config["taxonomic_profiling"]["bracken_read_length"],
        threshold=config["taxonomic_profiling"]["bracken_threshold"],
        level=config["taxonomic_profiling"]["bracken_level"],
    shell:
        """
        bracken -d {params.db} \
            -i {input.report} \
            -o {output.bracken} \
            -r {params.read_len} \
            -l {params.level} \
            -t {params.threshold} > {log} 2>&1
        """

# ==============================================================================
# Step 6: Build standardized sample x species abundance matrix
# ==============================================================================
rule build_abundance_matrix:
    input:
        bracken=expand(f"{OUT}/bracken/{{sample}}.bracken", sample=SAMPLES)
    output:
        matrix=f"{OUT}/matrix/species_abundance_matrix.csv"
    log:
        f"{LOG}/build_matrix.log"
    conda:
        "envs/qc.yaml"   # just needs pandas
    params:
        metric=config["matrix"]["abundance_metric"],
        min_prev=config["matrix"]["min_prevalence"],
    shell:
        """
        python scripts/build_abundance_matrix.py \
            --bracken-files {input.bracken} \
            --metric {params.metric} \
            --min-prevalence {params.min_prev} \
            --output {output.matrix} > {log} 2>&1
        """

# ==============================================================================
# Step 7: QC summary table (read counts at each stage, % host, % classified)
# ==============================================================================
rule qc_summary:
    input:
        cutadapt=expand(f"{OUT}/trimmed/{{sample}}.cutadapt.json", sample=SAMPLES),
        bowtie2=expand(f"{OUT}/host_removed/{{sample}}.bowtie2.stats.txt", sample=SAMPLES),
        kraken=expand(f"{OUT}/kraken2/{{sample}}.kreport", sample=SAMPLES),
    output:
        summary=f"{OUT}/qc/qc_summary.tsv"
    log:
        f"{LOG}/qc_summary.log"
    conda:
        "envs/qc.yaml"
    shell:
        """
        python scripts/build_qc_summary.py \
            --samples {SAMPLES} \
            --output-dir {OUT} \
            --output {output.summary} > {log} 2>&1
        """

# ==============================================================================
# Step 8 (optional): Functional profiling — HUMAnN3
# ==============================================================================
if config["functional_profiling"]["enabled"]:

    rule concat_for_humann:
        input:
            r1=f"{OUT}/host_removed/{{sample}}_R1.clean.fastq.gz",
            r2=f"{OUT}/host_removed/{{sample}}_R2.clean.fastq.gz",
        output:
            cat=f"{OUT}/host_removed/{{sample}}.concat.fastq.gz"
        shell:
            "cat {input.r1} {input.r2} > {output.cat}"

    rule humann3:
        input:
            f"{OUT}/host_removed/{{sample}}.concat.fastq.gz"
        output:
            genefamilies=f"{OUT}/humann3/{{sample}}_genefamilies.tsv",
            pathabundance=f"{OUT}/humann3/{{sample}}_pathabundance.tsv",
        log:
            f"{LOG}/humann3/{{sample}}.log"
        threads: config["functional_profiling"]["threads"]
        conda:
            "envs/functional.yaml"
        params:
            nt_db=config["functional_profiling"]["nucleotide_db"],
            prot_db=config["functional_profiling"]["protein_db"],
            outdir=f"{OUT}/humann3",
        shell:
            """
            humann --input {input} --output {params.outdir} \
                --threads {threads} \
                --nucleotide-database {params.nt_db} \
                --protein-database {params.prot_db} \
                --output-basename {wildcards.sample} > {log} 2>&1
            """

    rule build_pathway_matrix:
        input:
            expand(f"{OUT}/humann3/{{sample}}_pathabundance.tsv", sample=SAMPLES)
        output:
            f"{OUT}/matrix/pathway_abundance_matrix.csv"
        conda:
            "envs/qc.yaml"
        shell:
            """
            python scripts/build_pathway_matrix.py \
                --humann-files {input} \
                --output {output}
            """

# ==============================================================================
# Step 5 (alternative): Taxonomic profiling — MetaPhlAn
# ==============================================================================
# Active when taxonomic_profiling.profiler == "metaphlan". Both the MetaPhlAn and
# the Kraken2/Bracken builders target the canonical species_abundance_matrix.csv,
# so a ruleorder selects the active one and keeps the DAG unambiguous. MetaPhlAn
# accepts FASTA (.fna) reads directly via --input_type (set from input_format),
# giving an end-to-end path for quality-less FASTA input.
if PROFILER == "metaphlan":
    ruleorder: build_metaphlan_matrix > build_abundance_matrix
else:
    ruleorder: build_abundance_matrix > build_metaphlan_matrix

rule metaphlan:
    input:
        r1=f"{OUT}/host_removed/{{sample}}_R1.clean.fastq.gz",
        r2=f"{OUT}/host_removed/{{sample}}_R2.clean.fastq.gz",
    output:
        profile=f"{OUT}/metaphlan/{{sample}}_metaphlan.tsv",
    log:
        f"{LOG}/metaphlan/{{sample}}.log"
    threads: config["taxonomic_profiling"]["threads"]
    conda:
        "envs/taxonomy.yaml"
    params:
        db=config["taxonomic_profiling"].get("metaphlan_db", "resources/metaphlan_db"),
        index=config["taxonomic_profiling"].get("metaphlan_index", "latest"),
        input_type=("fasta" if INPUT_FORMAT == "fasta" else "fastq"),
        bt2out=f"{OUT}/metaphlan/{{sample}}.bowtie2.bz2",
    shell:
        """
        metaphlan {input.r1},{input.r2} \
            --input_type {params.input_type} \
            --bowtie2db {params.db} \
            --index {params.index} \
            --nproc {threads} \
            --bowtie2out {params.bt2out} \
            -o {output.profile} > {log} 2>&1
        """

rule build_metaphlan_matrix:
    input:
        expand(f"{OUT}/metaphlan/{{sample}}_metaphlan.tsv", sample=SAMPLES)
    output:
        matrix=f"{OUT}/matrix/species_abundance_matrix.csv"
    log:
        f"{LOG}/build_metaphlan_matrix.log"
    conda:
        "envs/qc.yaml"
    shell:
        """
        python scripts/build_metaphlan_matrix.py \
            --metaphlan-files {input} \
            --output {output.matrix} > {log} 2>&1
        """
