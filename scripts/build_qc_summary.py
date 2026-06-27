#!/usr/bin/env python3
"""
build_qc_summary.py
=====================
Aggregates per-sample stats from across the pipeline into one QC summary
table: raw read count, trimmed read count, % reads removed as host,
% reads classified by Kraken2.

This feeds the "Sample & Data Quality Summary" section of the final
clinical report (see project spec, section 4).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


def parse_cutadapt_json(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    read_counts = data.get("read_counts", {})
    return {
        "raw_reads": read_counts.get("input", None),
        "trimmed_reads": read_counts.get("output", None),
    }


def parse_bowtie2_stats(path: Path) -> dict:
    """
    Bowtie2 writes a human-readable alignment summary to stderr, which we've
    redirected to this file. We pull the overall alignment rate out of it;
    that rate = % of (trimmed) reads that mapped to the host genome and
    were therefore removed from the microbial read set.
    """
    text = path.read_text()
    match = re.search(r"([\d.]+)%\s+overall alignment rate", text)
    host_pct = float(match.group(1)) if match else None
    return {"pct_reads_host": host_pct}


def parse_kraken_report(path: Path) -> dict:
    """
    Kraken2 report: column 1 = % of reads, column 4 = rank code, root line
    (rank 'R') and 'U' (unclassified) rows give us classification rate.
    """
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["pct", "reads_clade", "reads_direct", "rank", "taxid", "name"],
    )
    unclassified_row = df[df["name"].str.strip() == "unclassified"]
    pct_unclassified = (
        float(unclassified_row["pct"].iloc[0]) if not unclassified_row.empty else None
    )
    pct_classified = 100.0 - pct_unclassified if pct_unclassified is not None else None
    return {"pct_classified": pct_classified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, help="Pipeline results/ dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    rows = []

    for sample in args.samples:
        row = {"sample": sample}

        cutadapt_path = out / "trimmed" / f"{sample}.cutadapt.json"
        if cutadapt_path.exists():
            row.update(parse_cutadapt_json(cutadapt_path))

        bowtie2_path = out / "host_removed" / f"{sample}.bowtie2.stats.txt"
        if bowtie2_path.exists():
            row.update(parse_bowtie2_stats(bowtie2_path))

        kraken_path = out / "kraken2" / f"{sample}.kreport"
        if kraken_path.exists():
            row.update(parse_kraken_report(kraken_path))

        rows.append(row)

    summary = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote QC summary for {len(rows)} samples to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
