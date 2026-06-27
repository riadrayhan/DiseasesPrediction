#!/usr/bin/env python3
"""
build_pathway_matrix.py
=======================
Combines per-sample HUMAnN3 ``*_pathabundance.tsv`` files into a single,
standardized sample x pathway matrix (CSV), mirroring
``build_abundance_matrix.py`` but for functional (pathway) features.

Only *unstratified* pathway rows are kept (HUMAnN emits both community-level
rows and per-species ``pathway|g__...s__...`` stratified rows; the latter are
dropped here so columns are one-per-pathway). Like the species matrix, this is
pure Python/pandas and needs no HUMAnN installation to run — it just parses the
text output.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def sample_name_from_path(path: Path) -> str:
    """sample01_pathabundance.tsv -> sample01"""
    return path.name.replace("_pathabundance.tsv", "").replace(".tsv", "")


def load_one_humann_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] < 2:
        raise ValueError(f"{path} does not look like a HUMAnN pathabundance file")
    pathway_col = df.columns[0]
    value_col = df.columns[1]
    df = df.rename(columns={pathway_col: "pathway", value_col: "abundance"})
    # Keep community-level (unstratified) rows only.
    df = df[~df["pathway"].astype(str).str.contains(r"\|")]
    df["sample"] = sample_name_from_path(path)
    return df[["sample", "pathway", "abundance"]]


def build_matrix(humann_files, drop_special: bool) -> pd.DataFrame:
    frames = [load_one_humann_file(Path(f)) for f in humann_files]
    if not frames:
        raise ValueError("No HUMAnN files provided.")
    combined = pd.concat(frames, ignore_index=True)

    if drop_special:
        special = {"UNMAPPED", "UNINTEGRATED"}
        combined = combined[~combined["pathway"].isin(special)]

    matrix = combined.pivot_table(
        index="sample",
        columns="pathway",
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
        "--humann-files", nargs="+", required=True,
        help="Paths to per-sample *_pathabundance.tsv files",
    )
    parser.add_argument(
        "--keep-special", action="store_true",
        help="Keep UNMAPPED / UNINTEGRATED rows (dropped by default)",
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    matrix = build_matrix(args.humann_files, drop_special=not args.keep_special)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output)
    print(
        f"Wrote {matrix.shape[0]} samples x {matrix.shape[1]} pathways "
        f"to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
