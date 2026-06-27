import numpy as np

from microbiome_predict import classify, data


def test_classifier_identifies_known_species():
    ref = classify.default_reference()
    clf = classify.KmerClassifier(ref)
    species = "Fusobacterium_nucleatum"
    read = ref[species][100:260]  # a fragment of that species' reference
    assert clf.classify_read(read) == species


def test_classify_read_unknown_returns_none():
    clf = classify.KmerClassifier(classify.default_reference())
    rng = np.random.default_rng(123)
    random_seq = "".join(rng.choice(list("ACGT"), size=160))
    assert clf.classify_read(random_seq) is None


def test_classify_profile_dominant_species():
    ref = classify.default_reference()
    clf = classify.KmerClassifier(ref)
    reads = classify.simulate_reads(ref, {"Bacteroides_fragilis": 1.0},
                                    n_reads=120, seed=2)
    profile, stats = clf.classify_profile(reads, "s")
    assert stats["n_classified"] > 90
    assert profile.iloc[0].idxmax() == "Bacteroides_fragilis"
    assert abs(profile.to_numpy().sum() - 1.0) < 1e-9


def test_simulate_samples_shape_and_classes():
    X, y = classify.simulate_samples(n_per_class=10)
    assert X.shape == (30, len(classify.DEMO_SPECIES))
    assert set(y.unique()) == {"healthy", "colorectal_cancer", "ibd"}


def test_end_to_end_raw_reads_to_crc_prediction():
    clf, bundle = classify.build_demo_classifier_and_model()
    ref = clf.reference
    abundances = {sp: 1.0 for sp in clf.species}
    abundances["Fusobacterium_nucleatum"] = 8.0
    abundances["Peptostreptococcus_anaerobius"] = 6.0
    abundances["Parvimonas_micra"] = 5.0
    abundances["Faecalibacterium_prausnitzii"] = 0.2
    reads = classify.simulate_reads(ref, abundances, n_reads=400, seed=3)

    profile, _ = clf.classify_profile(reads, "patient")
    X = data.align_features(profile, bundle.feature_names)
    assert str(bundle.classifier.predict(X)[0]) == "colorectal_cancer"
