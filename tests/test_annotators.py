import numpy as np
import pandas as pd

from celltyping.methods import (
    UNASSIGNED,
    annotate_clusters,
    annotate_flat,
    annotate_hierarchical,
    assign,
    score_sets,
    smooth_labels,
    summarize_levels,
)


def _accuracy(truth, pred_ids):
    return float(np.mean(np.asarray(truth) == np.asarray(pred_ids)))


def test_score_sets_and_assign(toy_adata, toy_tree):
    sets = toy_tree.marker_sets(["N_EX", "G_AS"])
    s = score_sets(toy_adata, sets)
    assert list(s.columns) == ["N_EX", "G_AS"] and s.shape[0] == toy_adata.n_obs
    ex = toy_adata.obs["truth"] == "N_EX"
    assert s.loc[ex, "N_EX"].mean() > s.loc[~ex, "N_EX"].mean()
    res = assign(s, min_score=0.0, min_margin=0.0)
    assert set(res.columns) == {"label", "score", "runner_up", "margin"}
    res2 = assign(s, min_score=99)
    assert (res2["label"] == UNASSIGNED).all()


def test_flat_annotation(toy_adata, toy_tree):
    annotate_flat(toy_adata, toy_tree, level="leaves")
    assert _accuracy(toy_adata.obs["truth"], toy_adata.obs["flat_id"]) > 0.9
    assert "flat_scores" in toy_adata.obsm and toy_adata.obsm["flat_scores"].shape[1] == 4


def test_hierarchical_annotation(toy_adata, toy_tree):
    annotate_hierarchical(toy_adata, toy_tree, min_cells=10)
    obs = toy_adata.obs
    assigned = obs["hier_id"] != "CL:0000000"
    assert assigned.mean() > 0.8
    # cells that reached a leaf are the right leaf; cells that stopped at an internal node are not wrong
    at_leaf = obs["hier_depth"] == 2
    assert at_leaf.mean() > 0.7
    assert _accuracy(obs.loc[at_leaf, "truth"], obs.loc[at_leaf, "hier_id"]) > 0.95
    # level-1 labels are coarse and consistent with truth
    coarse = obs["truth"].map(lambda l: "neuron" if l.startswith("N") else "glia")
    assert (obs.loc[assigned, "hier_level1"].astype(str) == coarse[assigned]).mean() > 0.98
    assert obs["hier_depth"].max() == 2
    lv = summarize_levels(toy_adata)
    assert set(lv["level"]) == {1, 2}
    sm = smooth_labels(toy_adata, "hier_label", k=10)
    assert sm.shape[0] == toy_adata.n_obs


def test_hierarchical_stops_at_parent_when_ambiguous(toy_adata, toy_tree):
    # remove sub-type markers of glia children from the data -> cells should stop at 'glia'
    kill = [g for n in ("G_AS", "G_OL") for g in toy_tree[n].markers if g != "ALDH1L1"]
    idx = [toy_adata.var_names.get_loc(g) for g in kill]
    X = toy_adata.X.copy()
    X[:, idx] = 0.0
    toy_adata.X = X
    annotate_hierarchical(toy_adata, toy_tree, min_cells=10, min_margin=1.0)
    reached_glia = toy_adata.obs["hier_id"].isin(["G", "G_AS", "G_OL"])
    assert reached_glia.sum() > 100
    assert (toy_adata.obs.loc[reached_glia, "hier_id"] == "G").mean() > 0.8


def test_cluster_annotation(toy_adata, toy_tree):
    tables = annotate_clusters(toy_adata, toy_tree, resolutions=(0.3,), level="leaves", min_score=0.0, min_margin=0.0)
    (k, tab), = tables.items()
    assert k == "cluster_r0.3"
    assert {"label", "score", "de_marker_overlap", "top_de"} <= set(tab.columns)
    assert f"{k}_label" in toy_adata.obs
    assert _accuracy(toy_adata.obs["truth"], toy_adata.obs[f"{k}_id"]) > 0.8
    assert isinstance(toy_adata.uns[f"{k}_de"], pd.DataFrame)
