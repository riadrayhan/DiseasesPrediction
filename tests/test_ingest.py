import gzip

import numpy as np
import pandas as pd

from microbiome_predict import ingest


def test_detect_kind():
    assert ingest.detect_kind("a.fna") == ingest.KIND_FASTA
    assert ingest.detect_kind("a.fasta.gz") == ingest.KIND_FASTA
    assert ingest.detect_kind("a.fastq") == ingest.KIND_FASTQ
    assert ingest.detect_kind("a.csv") == ingest.KIND_MATRIX
    assert ingest.detect_kind("a.tsv") == ingest.KIND_REPORT
    assert ingest.detect_kind("a.joblib") == ingest.KIND_MODEL
    # content sniffing when extension is unknown
    assert ingest.detect_kind("mystery", ">seq1\nACGT") == ingest.KIND_FASTA


def test_parse_fasta_and_qc():
    text = ">s1 desc\nACGT\nACGT\n>s2\nGGGGCC\n"
    records = ingest.parse_fasta(text)
    assert len(records) == 2
    assert records[0][1] == "ACGTACGT"  # multi-line sequence joined
    qc = ingest.sequence_qc(records, "sample")
    row = qc.iloc[0]
    assert row["n_sequences"] == 2
    assert row["total_bp"] == 14
    assert row["max_len"] == 8
    assert row["N50"] == 8
    # ACGTACGT has 4 GC, GGGGCC has 6 GC -> 10/14 bp.
    assert row["gc_percent"] == round(100 * (4 + 6) / 14, 2)


def test_parse_fastq():
    text = "@r1\nACGTAC\n+\nIIIIII\n@r2\nTTTT\n+\nIIII\n"
    records = ingest.parse_fastq(text)
    assert [seq for _, seq in records] == ["ACGTAC", "TTTT"]


def test_species_from_header():
    assert ingest.species_from_header("ctg|s__Escherichia_coli") == "Escherichia_coli"
    assert ingest.species_from_header("NZ_123.1 Escherichia coli strain K12") == "Escherichia_coli"
    assert ingest.species_from_header("just a random contig 12345") is None


def test_profile_from_headers_normalized():
    records = [
        ("s__Bacteroides_fragilis", "ACGT"),
        ("s__Bacteroides_fragilis", "ACGT"),
        ("s__Escherichia_coli", "ACGT"),
    ]
    profile, assigned = ingest.profile_from_headers(records, "sample")
    assert assigned == 3
    assert abs(profile.to_numpy().sum() - 1.0) < 1e-9
    assert abs(profile.loc["sample", "Bacteroides_fragilis"] - 2 / 3) < 1e-9


def test_metaphlan_report_uses_relative_abundance_not_taxid():
    text = (
        "#clade_name\tNCBI_tax_id\trelative_abundance\n"
        "k__Bacteria\t2\t100.0\n"
        "k__Bacteria|s__Escherichia_coli\t562\t70.0\n"
        "k__Bacteria|s__Bacteroides_fragilis\t817\t30.0\n"
    )
    frame = ingest.load_taxonomic_report(text, "sample")
    # Must use the abundance (70/30), not the tax ids (562/817).
    assert abs(frame.loc["sample", "Escherichia_coli"] - 0.7) < 1e-6
    assert abs(frame.loc["sample", "Bacteroides_fragilis"] - 0.3) < 1e-6


def test_ingest_text_routes_fasta():
    text = ">c1 s__Faecalibacterium_prausnitzii\nACGTACGT\n>c2 s__Faecalibacterium_prausnitzii\nACGT\n"
    result = ingest.ingest_text("sample.fna", text)
    assert result.kind == ingest.KIND_FASTA
    assert result.qc is not None
    assert result.abundance is not None
    assert "Faecalibacterium_prausnitzii" in result.abundance.columns


def test_ingest_text_fasta_without_taxonomy_gives_qc_only():
    text = ">contig_1\nACGTACGTACGT\n>contig_2\nTTTTGGGG\n"
    result = ingest.ingest_text("reads.fna", text)
    assert result.kind == ingest.KIND_FASTA
    assert result.qc is not None
    assert result.abundance is None  # honest: no taxonomy -> no fabricated profile


def test_ingest_bytes_gzip():
    text = ">c1 s__Escherichia_coli\nACGT\n"
    data = gzip.compress(text.encode())
    result = ingest.ingest_bytes("sample.fna.gz", data)
    assert result.kind == ingest.KIND_FASTA
    assert result.abundance is not None


def test_ingest_csv_matrix():
    csv = "sample,Sp_A,Sp_B\ns1,0.5,0.5\ns2,0.2,0.8\n"
    result = ingest.ingest_text("m.csv", csv)
    assert result.kind == ingest.KIND_MATRIX
    assert result.abundance.shape == (2, 2)
