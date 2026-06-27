#!/usr/bin/env python3
"""
build_metaphlan_matrix.py
=========================
Combines per-sample MetaPhlAn (v3/v4) profile outputs into a standardized
sample x species relative-abundance matrix (CSV) — the MetaPhlAn alternative to
the Kraken2/Bracken matrix builder, producing the same downstream format.

MetaPhlAn output is a TSV whose clade names are pipe-delimited lineages; we keep
species-level rows (those containing ``s__`` but not ``t__`` strain rows) and
report their relative abundance. Like the other matrix builders this only parses
text and needs no MetaPhlAn install.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def sample_name_from_path(path: Path) -> str:
    return (
        path.name.replace("_metaphlan.tsv", "")
        .replace(".metaphlan.tsv", "")
        .replace("_profile.tsv", "")
        .replace(".tsv", "")
    )


def load_one_metaphlan_file(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        header_cols = None
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                if line.startswith("#clade_name") or line.startswith("clade_name"):
                    header_cols = line.lstrip("#").split("\t")
                continue
            parts = line.split("\t")
            clade = parts[0]
            # Species level only: contains s__ but not strain t__.
            if "s__" not in clade or "t__" in clade:
                continue
            # Relative abundance is column index 2 in standard MetaPhlAn output
            # (clade_name, NCBI_tax_id, relative_abundance, ...); fall back to col 1.
            try:
                abundance = float(parts[2])
            except (IndexError, ValueError):
                abundance = float(parts[1])
            species = clade.split("|")[-1].replace("s__", "")
            rows.append((species, abundance))

    df = pd.DataFrame(rows, columns=["species", "abundance"])
    df["sample"] = sample_name_from_path(path)
    return df


def build_matrix(metaphlan_files) -> pd.DataFrame:
    frames = [load_one_metaphlan_file(Path(f)) for f in metaphlan_files]
    if not frames:
        raise ValueError("No MetaPhlAn files provided.")
    combined = pd.concat(frames, ignore_index=True)
    matrix = combined.pivot_table(
        index="sample",
        columns="species",
        values="abundance",
        aggfunc="sum",
        fill_value=0.0,
    )
    col_order = matrix.sum(axis=0).sort_values(ascending=False).index
    matrix = matrix[col_order]
    matrix.index.name = "sample"
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metaphlan-files", nargs="+", required=True,
        help="Paths to per-sample MetaPhlAn profile TSVs",
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    matrix = build_matrix(args.metaphlan_files)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output)
    print(
        f"Wrote {matrix.shape[0]} samples x {matrix.shape[1]} species "
        f"to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
