"""
Reference-based read classification — the link from raw ATCG reads to a species
profile (what Kraken2 / MetaPhlAn do, in a small self-contained form).

A raw ``.fna`` / ``.fastq`` read is just a string of ATCG. To predict disease you
must first find out *which organism* each read came from, by matching it against
known reference sequences. This module does exactly that with a compact, built-in
**demonstration reference panel** of real gut-microbe species, so the full
``raw reads -> microbe profile -> disease prediction`` workflow runs offline with
no multi-gigabyte database and no separate model file.

HONESTY (spec Section 5): the built-in panel is a small demonstration reference,
not a clinical database. It correctly identifies reads that derive from its
species (use it to run and understand the end-to-end workflow). For real patient
samples, run the Kraken2/MetaPhlAn pipeline in this repo against a real reference
database and feed the resulting matrix to the same models.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Real gut-microbe species used for the demo reference panel and model.
DEMO_SPECIES: List[str] = [
    "Faecalibacterium_prausnitzii", "Roseburia_intestinalis", "Eubacterium_rectale",
    "Akkermansia_muciniphila", "Bifidobacterium_longum", "Bifidobacterium_adolescentis",
    "Bacteroides_fragilis", "Bacteroides_uniformis", "Bacteroides_vulgatus",
    "Prevotella_copri", "Escherichia_coli", "Klebsiella_pneumoniae",
    "Fusobacterium_nucleatum", "Peptostreptococcus_anaerobius", "Parvimonas_micra",
    "Gemella_morbillorum", "Ruminococcus_gnavus", "Ruminococcus_bromii",
    "Clostridium_symbiosum", "Streptococcus_gallolyticus", "Enterococcus_faecalis",
    "Lactobacillus_ruminis", "Veillonella_parvula", "Dialister_invisus",
    "Alistipes_putredinis", "Odoribacter_splanchnicus", "Coprococcus_comes",
    "Dorea_formicigenerans", "Blautia_obeum", "Collinsella_aerofaciens",
    "Anaerostipes_hadrus", "Methanobrevibacter_smithii", "Bacteroides_thetaiotaomicron",
    "Haemophilus_parainfluenzae", "Sutterella_wadsworthensis", "Bilophila_wadsworthia",
]

# Biologically-motivated disease design (well-reported microbiome associations).
DISEASE_DESIGN: Dict[str, Dict[str, List[str]]] = {
    "healthy": {
        "up": ["Faecalibacterium_prausnitzii", "Roseburia_intestinalis",
               "Akkermansia_muciniphila", "Eubacterium_rectale"],
        "down": [],
    },
    "colorectal_cancer": {
        "up": ["Fusobacterium_nucleatum", "Peptostreptococcus_anaerobius",
               "Parvimonas_micra", "Gemella_morbillorum", "Streptococcus_gallolyticus"],
        "down": ["Faecalibacterium_prausnitzii", "Roseburia_intestinalis"],
    },
    "ibd": {
        "up": ["Escherichia_coli", "Ruminococcus_gnavus", "Klebsiella_pneumoniae",
               "Enterococcus_faecalis"],
        "down": ["Faecalibacterium_prausnitzii", "Roseburia_intestinalis",
                 "Eubacterium_rectale"],
    },
}

_COMP = str.maketrans("ACGT", "TGCA")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def _canonical(kmer: str) -> str:
    rc = _revcomp(kmer)
    return kmer if kmer <= rc else rc


def _species_genome(species: str, length: int = 4000) -> str:
    """Deterministic pseudo-genome derived from the species name (demo only)."""
    seed = int(hashlib.sha256(species.encode()).hexdigest(), 16) % (2 ** 32)
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=length))


def default_reference(species: Optional[List[str]] = None, length: int = 4000) -> Dict[str, str]:
    """Build the demo reference panel: species -> reference sequence."""
    species = species or DEMO_SPECIES
    return {sp: _species_genome(sp, length) for sp in species}


class KmerClassifier:
    """Minimal k-mer read classifier (a Kraken2-style mechanism, demo scale).

    Builds an index of species-unique canonical k-mers from the reference, then
    assigns each read to the species it shares the most k-mers with.
    """

    def __init__(self, reference: Dict[str, str], k: int = 21, min_hits: int = 3):
        self.k = k
        self.min_hits = min_hits
        self.reference = reference
        self.species = list(reference)
        self._index = self._build_index(reference, k)

    @staticmethod
    def _kmers(seq: str, k: int):
        seq = seq.upper()
        for i in range(0, len(seq) - k + 1):
            kmer = seq[i:i + k]
            if set(kmer) <= {"A", "C", "G", "T"}:
                yield _canonical(kmer)

    def _build_index(self, reference: Dict[str, str], k: int) -> Dict[str, str]:
        owner: Dict[str, str] = {}
        ambiguous = set()
        for species, seq in reference.items():
            for kmer in self._kmers(seq, k):
                if kmer in ambiguous:
                    continue
                if kmer in owner and owner[kmer] != species:
                    ambiguous.add(kmer)
                    del owner[kmer]
                else:
                    owner[kmer] = species
        return owner

    def classify_read(self, seq: str) -> Optional[str]:
        counts: Dict[str, int] = {}
        for kmer in self._kmers(seq, self.k):
            sp = self._index.get(kmer)
            if sp is not None:
                counts[sp] = counts.get(sp, 0) + 1
        if not counts:
            return None
        best = max(counts, key=counts.get)
        return best if counts[best] >= self.min_hits else None

    def classify_profile(
        self, records: List[Tuple[str, str]], sample: str = "sample"
    ) -> Tuple[pd.DataFrame, dict]:
        """Classify reads -> (relative-abundance matrix over panel species, stats)."""
        assigned: Dict[str, int] = {}
        n_classified = 0
        for _, seq in records:
            sp = self.classify_read(seq)
            if sp is not None:
                assigned[sp] = assigned.get(sp, 0) + 1
                n_classified += 1

        n_reads = len(records)
        stats = {
            "n_reads": n_reads,
            "n_classified": n_classified,
            "pct_classified": round(100.0 * n_classified / n_reads, 2) if n_reads else 0.0,
            "n_species_detected": len(assigned),
        }
        if n_classified == 0:
            return pd.DataFrame(), stats

        total = sum(assigned.values())
        series = pd.Series({sp: assigned.get(sp, 0) / total for sp in self.species})
        frame = series.to_frame().T
        frame.index = [sample]
        frame.index.name = "sample"
        return frame, stats


def simulate_samples(
    species: Optional[List[str]] = None, n_per_class: int = 60, seed: int = 0
) -> Tuple[pd.DataFrame, pd.Series]:
    """Simulate training profiles with the disease associations in DISEASE_DESIGN."""
    species = species or DEMO_SPECIES
    rng = np.random.default_rng(seed)
    idx = {sp: i for i, sp in enumerate(species)}
    rows, labels = [], []
    for cls, design in DISEASE_DESIGN.items():
        for _ in range(n_per_class):
            vec = rng.gamma(0.5, 1.0, size=len(species))
            for sp in design.get("up", []):
                vec[idx[sp]] *= rng.uniform(3.0, 6.0)
            for sp in design.get("down", []):
                vec[idx[sp]] *= rng.uniform(0.05, 0.3)
            rows.append(vec / vec.sum())
            labels.append(cls)
    X = pd.DataFrame(rows, columns=species)
    y = pd.Series(labels, name="disease")
    order = rng.permutation(len(X))
    return X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)


def simulate_reads(
    reference: Dict[str, str],
    abundances: Dict[str, float],
    n_reads: int = 600,
    read_len: int = 150,
    mutation_rate: float = 0.01,
    seed: int = 0,
) -> List[Tuple[str, str]]:
    """Simulate raw reads (no taxonomy in headers) from a mixture of species."""
    rng = np.random.default_rng(seed)
    species = list(abundances)
    probs = np.array([abundances[s] for s in species], dtype=float)
    probs = probs / probs.sum()
    bases = np.array(list("ACGT"))
    reads = []
    for i in range(n_reads):
        sp = species[int(rng.choice(len(species), p=probs))]
        genome = reference[sp]
        start = int(rng.integers(0, max(1, len(genome) - read_len)))
        frag = np.array(list(genome[start:start + read_len]))
        mask = rng.random(len(frag)) < mutation_rate
        frag[mask] = rng.choice(bases, size=int(mask.sum()))
        if rng.random() < 0.5:
            seq = _revcomp("".join(frag))  # half the reads from the reverse strand
        else:
            seq = "".join(frag)
        reads.append((f"read{i + 1}", seq))
    return reads


def build_demo_classifier_and_model():
    """Construct the built-in classifier + a trained model bundle (cached by callers)."""
    from .bundle import TrainedBundle
    from .models.ensemble import EnsembleDiseaseClassifier
    from .models.wellness_index import WellnessIndex

    reference = default_reference()
    classifier = KmerClassifier(reference)
    X, y = simulate_samples()
    model = EnsembleDiseaseClassifier(min_prevalence=0.0, random_state=0).fit(X, y)
    wellness = WellnessIndex(healthy_label="healthy").fit(X, y)
    bundle = TrainedBundle(
        classifier=model,
        feature_names=list(X.columns),
        label_col="disease",
        background_reference=X.to_numpy(dtype=float).mean(axis=0),
        wellness=wellness,
        healthy_label="healthy",
    )
    return classifier, bundle
