import json

from celltyping.knowledge.build import BuildParams, propagate_shared, prune, reweight_siblings
from celltyping.knowledge.markers import _parse_asctb_csv
from celltyping.knowledge.panel import load_panel
from celltyping.knowledge.tree import CellTypeTree


def test_tree_basic_ops(toy_tree):
    assert len(toy_tree) == 7
    assert set(toy_tree.leaves()) == {"N_EX", "N_IN", "G_AS", "G_OL"}
    assert toy_tree.path("G_AS") == ["CL:0000000", "G", "G_AS"]
    assert toy_tree.depth("G_AS") == 2
    toy_tree.reparent("G_OL", "N")
    assert "G_OL" in toy_tree["N"].children and "G_OL" not in toy_tree["G"].children
    toy_tree.remove("N", keep_children=True)
    assert toy_tree["N_EX"].parent == "CL:0000000"


def test_tree_roundtrip(toy_tree, tmp_path):
    p = toy_tree.save(tmp_path / "t.json")
    t2 = CellTypeTree.load(p)
    assert t2.to_dict() == toy_tree.to_dict()
    assert json.loads(p.read_text())["nodes"][0]["id"] == "CL:0000000"


def test_add_marker_accumulates_sources(toy_tree):
    toy_tree.add_marker("G_AS", "GFAP", 1.5, "other")
    m = toy_tree["G_AS"].markers["GFAP"]
    assert set(m.sources) == {"toy", "other"} and m.weight == 2.5
    toy_tree.add_marker("G_AS", "GFAP", 9.0, "toy")  # same source -> max, no double counting
    assert toy_tree["G_AS"].markers["GFAP"].weight == 9.0


def test_propagate_and_reweight(toy_tree):
    propagate_shared(toy_tree)  # ALDH1L1 is only in astrocyte -> nothing lifted; add shared gene to test
    toy_tree.add_marker("N_EX", "SYT1", 1.0, "toy")
    toy_tree.add_marker("N_IN", "SYT1", 1.0, "toy")
    propagate_shared(toy_tree)
    assert "SYT1" in toy_tree["N"].markers
    reweight_siblings(toy_tree)
    assert toy_tree["N_EX"].markers["SYT1"].weight == 0.5


def test_prune_by_panel(toy_tree):
    panel = {"SNAP25", "RBFOX3", "SYT1", "SLC17A7", "SATB2", "CUX2", "GAD1", "GAD2", "SST", "SOX9", "SOX10", "ALDH1L1", "GFAP", "AQP4"}
    for n in toy_tree.nodes.values():
        for m in n.markers.values():
            m.in_panel = m.gene in panel
    dropped = prune(toy_tree, BuildParams(min_markers=3), panel)
    ids = {d["id"] for d in dropped}
    assert "G_OL" in ids  # no oligodendrocyte markers in panel -> dropped
    # 'G' is a seed with a single child left: seeds are never collapsed
    assert "G_AS" in toy_tree and "G" in toy_tree and toy_tree["G_AS"].parent == "G"
    # a non-seed single-child node is collapsed
    toy_tree["G"].seed = False
    prune(toy_tree, BuildParams(min_markers=3), panel)
    assert "G" not in toy_tree and toy_tree["G_AS"].parent == "CL:0000000"


def test_prune_computational_markers_do_not_keep_nodes(toy_tree):
    # G_OL keeps its 3 curated markers; add a node backed only by CellGuide genes and a dead-end structural node
    toy_tree.add("X_CG", "cellguide-only type", "G", seed=True)
    for g in ("A1", "A2", "A3", "A4"):
        toy_tree.add_marker("X_CG", g, 1.0, "CellGuide")
    toy_tree.add("S_DEAD", "structural dead end", None, seed=False)
    for g in ("B1", "B2", "B3", "B4"):
        toy_tree.add_marker("S_DEAD", g, 1.0, "toy")
    dropped = prune(toy_tree, BuildParams(min_markers=3), None)
    reasons = {d["id"]: d["reason"] for d in dropped}
    assert "X_CG" in reasons and "curated" in reasons["X_CG"]
    assert "S_DEAD" in reasons and "without descendants" in reasons["S_DEAD"]
    assert "G_OL" in toy_tree and "G_AS" in toy_tree


def test_panel_from_xenium_json(tmp_path):
    d = {"payload": {"targets": [
        {"type": {"descriptor": "gene", "data": {"name": "Gfap"}}},
        {"type": {"descriptor": "negative_control", "data": {"name": "NegControlProbe_0001"}}},
        {"type": {"descriptor": "deprecated_codeword", "data": {"name": "Deprecated_1"}}},
        {"type": {"descriptor": "gene", "data": {"name": "AQP4"}}},
    ]}}
    p = tmp_path / "gene_panel.json"
    p.write_text(json.dumps(d))
    assert load_panel(p) == ["AQP4", "GFAP"]
    (tmp_path / "genes.txt").write_text("gene\nMBP\nPLP1\n")
    assert load_panel(tmp_path / "genes.txt") == ["MBP", "PLP1"]


def test_asctb_parser():
    csv = "\n".join([
        "v1.0,,,",
        "Author Name(s):,x,,",
        "AS/1,AS/1/LABEL,AS/1/ID,CT/1,CT/1/LABEL,CT/1/ID,CT/2,CT/2/LABEL,CT/2/ID,BGene/1,BGene/1/LABEL,BGene/1/ID,BGene/2,BGene/2/LABEL,BGene/2/ID",
        "brain,brain,UBERON:0000955,glial cell,glial cell,CL:0000125,astrocyte,astrocyte,CL:0000127,GFAP,GFAP,HGNC:4235,AQP4,AQP4,HGNC:637",
        "brain,brain,UBERON:0000955,neuron,neuron,CL:0000540,,,,RBFOX3,RBFOX3,HGNC:27097,,,",
    ])
    df = _parse_asctb_csv(csv, "allen-brain", "UBERON:0000955")
    assert set(df["cl_id"]) == {"CL:0000127", "CL:0000540"}
    assert set(df.loc[df["cl_id"] == "CL:0000127", "gene"]) == {"GFAP", "AQP4"}
    assert (df["source"] == "ASCT+B").all() and (df["weight"] == 1.5).all()
