"""Tests for the scanpy-style facade (``ct.pp`` / ``ct.tl`` / ``ct.pl`` / ``ct.bm`` / ``ct.annotate``) on toy data."""

import numpy as np
import pandas as pd
import pytest
import anndata as ad

import celltyping as ct

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _counts_adata(toy_adata):
    a = toy_adata.copy()
    a.X = np.expm1(a.X).round()  # back to integer counts
    return a


def test_namespaces_exist():
    for ns, names in {ct.pp: ["qc", "normalize", "knn_smooth", "is_raw_counts", "preprocess"],
                      ct.tl: ["knowledge", "hierarchical", "flat", "clusters", "score", "expected_cell_types"],
                      ct.pl: ["tree", "spatial", "composition", "scores", "confusion"],
                      ct.bm: ["reference", "compare", "agreement"],
                      ct.read: ["xenium", "cosmx", "h5ad"]}.items():
        for n in names:
            assert callable(getattr(ns, n))
    assert callable(ct.annotate)
    assert ct.settings.species == "human"


def test_is_raw_counts(toy_adata):
    assert not ct.pp.is_raw_counts(toy_adata)  # log1p floats
    a = _counts_adata(toy_adata)
    assert ct.pp.is_raw_counts(a)
    ct.pp.normalize(a)
    assert not ct.pp.is_raw_counts(a) and "counts" in a.layers


def test_preprocess_from_counts(toy_adata):
    a = ct.pp.preprocess(_counts_adata(toy_adata), min_counts=1, min_genes=1, knn_smooth_k=10)
    assert "knn_smooth" in a.layers and "counts" in a.layers and a.uns["normalized"]["log1p"]
    assert a.obs["qc_pass"].all() and (a.obs["qc_flag"] == "high_quality").all()


def test_qc_flag_propagates(toy_adata, toy_tree):
    a = _counts_adata(toy_adata)
    n0 = a.n_obs
    # make the first 30 cells low quality
    a.X[:30] = 0
    a.X[:30, :3] = 1  # 3 transcripts
    a = ct.pp.preprocess(a, min_counts=20, knn_smooth_k=10)
    assert a.n_obs == n0 and a.obs["qc_pass"].sum() == n0 - 30
    assert list(a.obs["qc_flag"][:30].unique()) == ["low_quality"]
    # low-quality rows are not smoothed (identical to X), high-quality ones are
    import numpy as np
    assert np.allclose(a.layers["knn_smooth"][:30], a.X[:30])
    assert not np.allclose(a.layers["knn_smooth"][30:60], a.X[30:60])
    ct.annotate(a, tree=toy_tree, min_cells=10, knn_smooth=None)
    assert (a.obs["hier_label"][:30].astype(str) == "low_quality").all()
    assert (a.obs["hier_id"][:30].astype(str) == "low_quality").all()
    assert a.obs["hier_score"][:30].isna().all() and a.obs["hier_score"][30:].notna().sum() > 0
    assert a.obsm["hier_scores"].iloc[:30].isna().all().all()
    assert a.uns["celltyping"]["n_low_quality"] == 30
    # benchmark ignores low-quality cells
    ref = ct.bm.reference(a, "truth", {l: l for l in ["N_EX", "N_IN", "G_AS", "G_OL"]}, toy_tree)
    summ = ct.bm.compare(a, ref, toy_tree, methods=("hier",))
    assert summ.loc["hier", "n_eval"] == n0 - 30
    ct.tl.flat(a, toy_tree, layer=None)
    assert (a.obs["flat_label"][:30].astype(str) == "low_quality").all()
    ct.tl.clusters(a, toy_tree, resolutions=(0.5,))
    assert (a.obs["cluster_r0.5_label"][:30].astype(str) == "low_quality").all()
    # filter=True removes them instead
    b = ct.pp.qc(_counts_adata(toy_adata), min_counts=1_000_000, filter=True)
    assert b.n_obs == 0
    # re-running with a different threshold recomputes the flag and overwrites the previous labels
    ct.annotate(a, tree=toy_tree, min_cells=10, knn_smooth=None, min_counts=1)
    assert a.obs["qc_pass"].all() and (a.obs["hier_label"].astype(str) != "low_quality").all()
    assert a.obs["hier_score"].notna().sum() > 0


def test_qc_xenium_control_counts_not_all_codewords(toy_adata):
    import anndata as ad
    import numpy as np
    a = ad.AnnData(X=np.array([[50, 0], [50, 0], [5, 0]], dtype=float),
                   obs=pd.DataFrame({
                       "transcript_counts": [369.0, 295.0, 5.0],
                       "control_probe_counts": [2, 2, 0],
                       "genomic_control_counts": [0, 0, 0],
                       "unassigned_codeword_counts": [58, 47, 0],
                       "deprecated_codeword_counts": [0, 0, 0],
                   }))
    # user mistake: summing every Xenium control column into control_counts over-flags high-count cells
    a.obs["control_counts"] = a.obs.filter(regex="control|unassigned|deprecated").sum(axis=1)
    ct.pp.qc(a, min_counts=10)
    assert a.obs["qc_pass"].tolist() == [True, True, False]
    assert a.obs["qc_reason"].tolist() == ["pass", "pass", "low_counts"]


def test_qc_duplicate_obs_names(toy_adata, toy_tree):
    a = _counts_adata(toy_adata)
    a.X[:10] = 0
    a.obs_names = ["c"] * a.n_obs  # pathological but must not misalign
    ct.pp.preprocess(a, min_counts=20, knn_smooth_k=None)
    ct.tl.flat(a, toy_tree, layer=None)
    assert (a.obs["flat_label"].astype(str).to_numpy()[:10] == "low_quality").all()
    assert (a.obs["flat_label"].astype(str).to_numpy()[10:] != "low_quality").all()


def test_annotate_with_prebuilt_tree(toy_adata, toy_tree):
    a = ct.annotate(_counts_adata(toy_adata), tree=toy_tree, min_cells=10, knn_smooth=10)
    assert "hier_label" in a.obs and a.uns["celltyping"]["method"] == "hierarchical"
    at_leaf = a.obs["hier_depth"] == 2
    assert (a.obs.loc[at_leaf, "hier_id"] == a.obs.loc[at_leaf, "truth"]).mean() > 0.95
    t2 = ct.tl.get_tree(a)
    assert set(t2.nodes) == set(toy_tree.nodes)
    # flat + cluster through the same entry point, no re-preprocessing (X is already log-normalised)
    ct.annotate(a, tree=toy_tree, method="flat")
    assert (a.obs["flat_id"] == a.obs["truth"]).mean() > 0.9
    ct.annotate(a, tree=toy_tree, method="cluster", resolutions=(0.5,), knn_smooth=None)
    assert "cluster_r0.5_label" in a.obs
    with pytest.raises(ValueError):
        ct.annotate(a, method="hierarchical")  # neither tissue nor tree


def test_tl_score_and_smooth(toy_adata, toy_tree):
    s = ct.tl.score(toy_adata, toy_tree, layer=None)
    assert s.shape == (toy_adata.n_obs, 4) and "excitatory" in s.columns
    ct.tl.hierarchical(toy_adata, toy_tree, min_cells=10, layer=None)
    sm = ct.tl.smooth_labels(toy_adata, "hier_label", k=10)
    assert sm.shape[0] == toy_adata.n_obs


def test_bm_and_pl(toy_adata, toy_tree, tmp_path):
    ct.tl.hierarchical(toy_adata, toy_tree, min_cells=10, layer=None)
    ct.tl.flat(toy_adata, toy_tree, layer=None)
    mapping = {l: l for l in ["N_EX", "N_IN", "G_AS", "G_OL"]}
    ref = ct.bm.reference(toy_adata, "truth", mapping, toy_tree)
    assert ref.notna().all() and "ref_id" in toy_adata.obs
    summ = ct.bm.compare(toy_adata, ref, toy_tree, methods=("hier", "flat"))
    assert set(summ.index) == {"hier", "flat"} and summ.loc["hier", "lineage_acc"] > 0.8
    conf = ct.bm.confusion(toy_adata, ref, toy_tree, key="hier")
    ax = ct.pl.confusion(conf, save=tmp_path / "conf.png")
    assert (tmp_path / "conf.png").exists()
    ct.pl.tree(toy_tree, save=tmp_path / "tree.png")
    ct.pl.spatial(toy_adata, "hier_label", tree=toy_tree, save=tmp_path / "sp.png")
    ct.pl.spatial(toy_adata, "hier_score")
    ct.pl.composition(toy_adata, tree=toy_tree)
    ct.pl.scores(toy_adata, tree=toy_tree)
    matplotlib.pyplot.close("all")
    assert ax is not None
