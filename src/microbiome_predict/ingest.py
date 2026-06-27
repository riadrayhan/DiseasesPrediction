"""
Multi-format input ingestion & support system.

Accepts the file types a user is likely to have and routes each to the right
handler, producing either an abundance matrix the predictive models can consume
or a deterministic QC summary:

* ``.fna`` / ``.fasta`` / ``.fa`` (FASTA reads or contigs)
* ``.fastq`` / ``.fq`` (FASTQ reads)
* ``.csv`` (sample x species abundance matrix)
* ``.tsv`` / ``.txt`` (taxonomic report: MetaPhlAn / Kraken-style / 2-column)
* ``.joblib`` (trained model bundle — handled by the caller)
* any of the above ``.gz``-compressed

Honesty note (spec Section 5): the only deterministic information derivable from
raw sequence is its **QC profile** (read counts, lengths, GC, N50). Turning raw
*unannotated* reads into a species table requires a taxonomic classifier
(Kraken2 / MetaPhlAn) + reference database — the upstream Snakemake pipeline.
This module will, however, build a real relative-abundance profile when the
sequence **headers already carry taxonomy** (e.g. NCBI binomial names or
``s__Genus_species`` lineage tags), which is common for reference ``.fna`` and
classifier exports. It never fabricates taxonomic assignments.
"""

from __future__ import annotations

import gzip
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

FASTA_EXT = {".fna", ".fasta", ".fa", ".ffn"}
FASTQ_EXT = {".fastq", ".fq"}
MATRIX_EXT = {".csv"}
REPORT_EXT = {".tsv", ".txt", ".report", ".mpa"}
MODEL_EXT = {".joblib", ".pkl"}

KIND_FASTA = "fasta"
KIND_FASTQ = "fastq"
KIND_MATRIX = "abundance_matrix"
KIND_REPORT = "taxonomic_report"
KIND_MODEL = "model_bundle"
KIND_UNKNOWN = "unknown"

_SPECIES_RE = re.compile(r"s__([A-Za-z0-9_.\-\[\]]+)")
_BINOMIAL_RE = re.compile(r"\b([A-Z][a-z]+)[ _]([a-z]{2,})\b")


@dataclass
class IngestResult:
    filename: str
    kind: str
    abundance: Optional[pd.DataFrame] = None  # samples x species (relative)
    qc: Optional[pd.DataFrame] = None         # one-row QC summary
    n_records: int = 0
    n_assigned: int = 0                        # records mapped to a taxon
    message: str = ""


# ---------------------------------------------------------------------------
# Decompression / decoding
# ---------------------------------------------------------------------------
def decode_bytes(data: bytes) -> str:
    """Decode raw bytes to text, transparently gunzipping if needed."""
    if data[:2] == b"\x1f\x8b":  # gzip magic number
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="replace")


def _strip_gz(name: str) -> str:
    return name[:-3] if name.lower().endswith(".gz") else name


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------
def detect_kind(filename: str, head: str = "") -> str:
    ext = Path(_strip_gz(filename)).suffix.lower()
    if ext in MODEL_EXT:
        return KIND_MODEL
    if ext in FASTQ_EXT:
        return KIND_FASTQ
    if ext in FASTA_EXT:
        return KIND_FASTA
    if ext in MATRIX_EXT:
        return KIND_MATRIX
    if ext in REPORT_EXT:
        return KIND_REPORT
    stripped = head.lstrip()
    if stripped.startswith(">"):
        return KIND_FASTA
    if stripped.startswith("@"):
        return KIND_FASTQ
    return KIND_UNKNOWN


# ---------------------------------------------------------------------------
# Sequence parsing
# ---------------------------------------------------------------------------
def parse_fasta(text: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    parts: List[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header = line[1:].strip()
            parts = []
        elif line.strip():
            parts.append(line.strip())
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def parse_fastq(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    records: List[Tuple[str, str]] = []
    n = (len(lines) // 4) * 4
    for i in range(0, n, 4):
        if lines[i].startswith("@"):
            records.append((lines[i][1:].strip(), lines[i + 1].strip()))
    return records


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------
def _n50(lengths: np.ndarray) -> int:
    if lengths.size == 0:
        return 0
    ordered = np.sort(lengths)[::-1]
    half = ordered.sum() / 2.0
    idx = int(np.searchsorted(np.cumsum(ordered), half))
    return int(ordered[min(idx, ordered.size - 1)])


def sequence_qc(records: List[Tuple[str, str]], sample: str) -> pd.DataFrame:
    lengths = np.array([len(seq) for _, seq in records], dtype=float)
    total = float(lengths.sum())
    gc = sum(seq.upper().count("G") + seq.upper().count("C") for _, seq in records)
    stats = {
        "sample": sample,
        "n_sequences": int(lengths.size),
        "total_bp": int(total),
        "min_len": int(lengths.min()) if lengths.size else 0,
        "mean_len": round(float(lengths.mean()), 1) if lengths.size else 0.0,
        "max_len": int(lengths.max()) if lengths.size else 0,
        "N50": _n50(lengths),
        "gc_percent": round(100.0 * gc / total, 2) if total else 0.0,
    }
    return pd.DataFrame([stats])


# ---------------------------------------------------------------------------
# Taxonomy from sequence headers (annotation-derived, never fabricated)
# ---------------------------------------------------------------------------
def species_from_header(header: str) -> Optional[str]:
    match = _SPECIES_RE.search(header)
    if match:
        # Keep the underscore convention (e.g. Escherichia_coli), matching the
        # MetaPhlAn/Bracken matrix builders so profiles align with trained models.
        return match.group(1).strip()
    # Drop a leading accession token, then look for a Genus species binomial.
    desc = header.split(None, 1)
    text = desc[1] if len(desc) > 1 else header
    match = _BINOMIAL_RE.search(text)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None


def profile_from_headers(
    records: List[Tuple[str, str]], sample: str, weight: str = "reads"
) -> Tuple[pd.DataFrame, int]:
    """Relative-abundance profile from taxonomy embedded in headers.

    ``weight='reads'`` counts sequences per taxon; ``weight='bp'`` weights by
    sequence length (sensible for contigs/assemblies).
    """
    counts: dict[str, float] = {}
    assigned = 0
    for header, seq in records:
        species = species_from_header(header)
        if species is None:
            continue
        counts[species] = counts.get(species, 0.0) + (len(seq) if weight == "bp" else 1.0)
        assigned += 1
    if not counts:
        return pd.DataFrame(), 0
    series = pd.Series(counts, dtype=float)
    series = series / series.sum()
    frame = series.to_frame().T
    frame.index = [sample]
    frame.index.name = "sample"
    return frame, assigned


# ---------------------------------------------------------------------------
# Tabular taxonomic reports
# ---------------------------------------------------------------------------
def load_taxonomic_report(text: str, sample: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]

    # MetaPhlAn / lineage style: keep species rows (s__ and not strain t__).
    species: dict[str, float] = {}
    for line in lines:
        parts = line.split("\t")
        name = parts[0]
        if "s__" in name and "t__" not in name:
            value = _last_float(parts[1:])
            if value is not None:
                key = name.split("|")[-1].replace("s__", "")
                species[key] = species.get(key, 0.0) + value
    if species:
        return _series_to_matrix(pd.Series(species, dtype=float), sample)

    # Generic 2+ column "name<sep>value".
    rows: dict[str, float] = {}
    for line in lines:
        parts = re.split(r"[\t,]", line)
        if len(parts) >= 2:
            value = _last_float(parts[1:])
            if value is not None:
                rows[parts[0].strip()] = value
    if rows:
        return _series_to_matrix(pd.Series(rows, dtype=float), sample)
    return pd.DataFrame()


def _first_float(tokens) -> Optional[float]:
    for tok in tokens:
        try:
            return float(tok)
        except (TypeError, ValueError):
            continue
    return None


def _last_float(tokens) -> Optional[float]:
    for tok in reversed(list(tokens)):
        try:
            return float(tok)
        except (TypeError, ValueError):
            continue
    return None


def _series_to_matrix(series: pd.Series, sample: str) -> pd.DataFrame:
    if series.sum() > 0:
        series = series / series.sum()
    frame = series.to_frame().T
    frame.index = [sample]
    frame.index.name = "sample"
    return frame


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------
def ingest_text(filename: str, text: str) -> IngestResult:
    sample = Path(_strip_gz(filename)).stem
    kind = detect_kind(filename, text[:256])

    if kind == KIND_FASTA or kind == KIND_FASTQ:
        records = parse_fasta(text) if kind == KIND_FASTA else parse_fastq(text)
        qc = sequence_qc(records, sample)
        weight = "bp" if (records and np.mean([len(s) for _, s in records]) > 500) else "reads"
        abundance, assigned = profile_from_headers(records, sample, weight=weight)
        if abundance.empty:
            message = (
                f"Parsed {len(records)} sequences. No taxonomy was found in the "
                "headers, so no species profile could be derived. Raw reads must "
                "be classified by the Kraken2/MetaPhlAn pipeline to obtain "
                "abundances; the QC summary above is the deterministic result."
            )
        else:
            message = (
                f"Parsed {len(records)} sequences; derived a {abundance.shape[1]}-"
                f"species profile from header taxonomy ({assigned} sequences "
                f"assigned, weighted by {weight})."
            )
        return IngestResult(filename, kind, abundance=(None if abundance.empty else abundance),
                            qc=qc, n_records=len(records), n_assigned=assigned, message=message)

    if kind == KIND_MATRIX:
        frame = pd.read_csv(io.StringIO(text), index_col=0)
        frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return IngestResult(filename, KIND_MATRIX, abundance=frame,
                            n_records=frame.shape[0],
                            message=f"Loaded abundance matrix: {frame.shape[0]} samples x "
                                    f"{frame.shape[1]} features.")

    if kind == KIND_REPORT:
        frame = load_taxonomic_report(text, sample)
        if frame.empty:
            # Fall back to treating it as a matrix.
            try:
                frame = pd.read_csv(io.StringIO(text), sep="\t", index_col=0)
                frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                return IngestResult(filename, KIND_MATRIX, abundance=frame,
                                    n_records=frame.shape[0],
                                    message=f"Loaded TSV matrix: {frame.shape}.")
            except Exception:
                return IngestResult(filename, KIND_REPORT,
                                    message="Could not parse a taxonomic profile from this file.")
        return IngestResult(filename, KIND_REPORT, abundance=frame, n_records=1,
                            message=f"Parsed taxonomic report: {frame.shape[1]} species.")

    return IngestResult(filename, KIND_UNKNOWN,
                        message="Unrecognized file type. Supported: .fna/.fasta/.fastq, "
                                ".csv, .tsv/.txt (taxonomic report), .joblib (model).")


def ingest_bytes(filename: str, data: bytes) -> IngestResult:
    return ingest_text(filename, decode_bytes(data))


def ingest_path(path: str | Path) -> IngestResult:
    p = Path(path)
    return ingest_bytes(p.name, p.read_bytes())
