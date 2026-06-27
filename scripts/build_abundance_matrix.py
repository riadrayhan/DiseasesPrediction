#!/usr/bin/env python3
"""
build_abundance_matrix.py
==========================
Parses per-sample Bracken output files and combines them into a single,
standardized sample x species relative-abundance matrix (CSV).

Bracken output columns (tab-separated):
    name  taxonomy_id  taxonomy_lvl  kraken_assigned_reads  added_reads
    new_est_reads  fraction_total_reads

Output:
    A CSV with samples as rows and species names as columns. Values are
    relative abundances (fractions, summing to ~1.0 per sample) unless
    --metric new_est_reads is chosen, in which case raw estimated read
    counts are reported instead.

This script has no dependency on Kraken2/Bracken being installed — it only
parses their text output — so it can be developed/tested independently of
the rest of the pipeline.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "name",
    "taxonomy_id",
    "taxonomy_lvl",
    "kraken_assigned_reads",
    "added_reads",
    "new_est_reads",
    "fraction_total_reads",
]


def sample_name_from_path(path: Path) -> str:
    """sample01.bracken -> sample01"""
    return path.name.replace(".bracken", "")


def load_one_bracken_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing expected Bracken columns: {sorted(missing)}"
        )
    df["sample"] = sample_name_from_path(path)
    return df


def build_matrix(bracken_files, metric: str, min_prevalence: float) -> pd.DataFrame:
    frames = [load_one_bracken_file(Path(f)) for f in bracken_files]
    if not frames:
        raise ValueError("No Bracken files provided.")

    combined = pd.concat(frames, ignore_index=True)

    # Pivot: rows = sample, columns = species name, values = chosen metric
    matrix = combined.pivot_table(
        index="sample",
        columns="name",
        values=metric,
        aggfunc="sum",
        fill_value=0.0,
    )

    # Prevalence filter: drop species present (non-zero) in too few samples
    if min_prevalence > 0:
        prevalence = (matrix > 0).mean(axis=0)
        keep_cols = prevalence[prevalence >= min_prevalence].index
        matrix = matrix[keep_cols]

    # Sort species by total abundance across samples, descending — easier to eyeball
    col_order = matrix.sum(axis=0).sort_values(ascending=False).index
    matrix = matrix[col_order]

    matrix.index.name = "sample"
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bracken-files", nargs="+", required=True,
        help="Paths to per-sample *.bracken files",
    )
    parser.add_argument(
        "--metric", choices=["fraction_total_reads", "new_est_reads"],
        default="fraction_total_reads",
        help="Which Bracken column to use as the abundance value",
    )
    parser.add_argument(
        "--min-prevalence", type=float, default=0.0,
        help="Drop species present in fewer than this fraction of samples (0-1)",
    )
    parser.add_argument(
        "--output", required=True, help="Output CSV path"
    )
    args = parser.parse_args()

    matrix = build_matrix(args.bracken_files, args.metric, args.min_prevalence)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output)

    print(
        f"Wrote {matrix.shape[0]} samples x {matrix.shape[1]} species "
        f"to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
