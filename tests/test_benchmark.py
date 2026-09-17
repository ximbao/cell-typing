import numpy as np
import pandas as pd

from celltyping.methods import annotate_hierarchical
from celltyping.benchmark import agreement, hierarchical_prf, marker_specificity, project_to_reference_classes, reference_ids, spatial_coherence
from celltyping.benchmark.harmonize import ANCESTOR, DESCENDANT, EXACT, UNASSIGNED_REL, WRONG, relation_table


def test_relations_and_projection(toy_tree):
    ref = pd.Series(["N_EX", "N_EX", "N", "G_AS", "G_AS", "G_OL"])
    pred = pd.Series(["N_EX", "N", "N_IN", "G", "N_EX", "unassigned"])
    rel = relation_table(pred, ref, toy_tree)
    assert list(rel) == [EXACT, ANCESTOR, DESCENDANT, ANCESTOR, WRONG, UNASSIGNED_REL]
    proj = project_to_reference_classes(pd.Series(["N_IN", "G", "CL:0000000"]), {"N", "G_AS"}, toy_tree)
    assert list(proj) == ["N", "G", "unassigned"]  # N_IN -> its reference ancestor N; G kept (coarser); root -> unassigned


def test_hierarchical_prf(toy_tree):
    ref = pd.Series(["N_EX", "G_AS"])
    assert hierarchical_prf(pd.Series(["N_EX", "G_AS"]), ref, toy_tree)["hF1"] == 1.0
    coarse = hierarchical_prf(pd.Series(["N", "G"]), ref, toy_tree)  # right lineage, less specific: perfect precision
    assert coarse["hP"] == 1.0 and coarse["hR"] == 0.5
    assert hierarchical_prf(pd.Series(["G_OL", "N_IN"]), ref, toy_tree)["hF1"] == 0.0


def test_reference_mapping_lifts_unknown_ids(toy_tree):
    labels = pd.Series(["Excitatory neurons", "Astro", "Junk", "Unassigned"])
    mapping = {"Excitatory neurons": "N_EX", "Astro": "G_AS", "Junk": "NOT_A_NODE"}
    ids = reference_ids(labels, mapping, toy_tree, ignore=["Unassigned"])
    assert list(ids[:2]) == ["N_EX", "G_AS"] and ids[2] is None and ids[3] is None


def test_agreement_end_to_end(toy_adata, toy_tree):
    annotate_hierarchical(toy_adata, toy_tree, min_cells=10)
    ref = toy_adata.obs["truth"].astype(object)
    pred = toy_adata.obs["hier_id"].astype(object)
    summ, per_class, conf = agreement(pred, ref, toy_tree)
    assert abs(summ["lineage_acc"] + summ["coarser"] + summ["unassigned"] + summ["wrong"] - 1) < 1e-9
    assert summ["lineage_acc"] > 0.6 and summ["wrong"] < 0.05
    assert set(per_class["ref_id"]) == {"N_EX", "N_IN", "G_AS", "G_OL"}
    assert per_class["f1"].notna().all()
    assert conf.to_numpy().sum() == len(ref)
    sc = spatial_coherence(np.asarray(toy_adata.obsm["spatial"]), pred, toy_tree, k=5)["spatial_coherence"]
    assert 0.5 < sc <= 1.0  # populations are spatially separated in the toy layout
    ms = marker_specificity(toy_adata, pred, toy_tree, top_k=10, min_cells=10)
    assert ms["marker_specificity"] > 0 and ms["marker_specificity_n_types"] >= 4
