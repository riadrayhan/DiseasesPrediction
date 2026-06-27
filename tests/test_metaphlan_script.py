import importlib.util
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metaphlan_matrix_parse(tmp_path):
    mod = _load_script("build_metaphlan_matrix")
    content = (
        "#clade_name\tNCBI_tax_id\trelative_abundance\n"
        "k__Bacteria\t2\t100.0\n"
        "k__Bacteria|p__Firmicutes|g__Faecalibacterium|s__Faecalibacterium_prausnitzii\t1\t60.0\n"
        "k__Bacteria|p__Bacteroidetes|g__Bacteroides|s__Bacteroides_fragilis\t2\t40.0\n"
        "k__Bacteria|p__Bacteroidetes|g__Bacteroides|s__Bacteroides_fragilis|t__strain\t3\t10.0\n"
    )
    f = tmp_path / "S1_metaphlan.tsv"
    f.write_text(content)

    matrix = mod.build_matrix([str(f)])
    assert list(matrix.index) == ["S1"]
    assert "Faecalibacterium_prausnitzii" in matrix.columns
    assert "Bacteroides_fragilis" in matrix.columns
    # Strain-level (t__) rows are excluded, so fragilis stays at 40.0.
    assert abs(matrix.loc["S1", "Bacteroides_fragilis"] - 40.0) < 1e-9
