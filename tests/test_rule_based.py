"""Tests for rule-based panel annotation."""

import numpy as np
import pandas as pd

import celltyping as ct
from celltyping.methods.rule_based import annotate_rule_based, panels_from_tree


def _counts_adata(toy_adata):
    a = toy_adata.copy()
    a.X = np.expm1(a.X).round()
    return a


def _accuracy(truth, pred):
    t = truth.astype(str).to_numpy()
    p = pred.astype(str).to_numpy()
    return float((t == p).mean())


def test_panels_from_tree(toy_tree):
    p = panels_from_tree(toy_tree, n_markers=3)
    assert set(p) == set(toy_tree.leaves())
    assert all(len(g) <= 3 for g in p.values())


def test_rule_based_all_genes_required(toy_adata, toy_tree):
    a = _counts_adata(toy_adata)
    ct.pp.preprocess(a, min_counts=1, knn_smooth_k=None)
    annotate_rule_based(a, toy_tree, n_markers=3)
    assert "rule_label" in a.obs and a.obsm["rule_scores"].shape[1] == 4
    gi = a.var_names.get_loc("GFAP")
    ast = a.obs["truth"] == "G_AS"
    a.layers["counts"][ast, gi] = 0
    annotate_rule_based(a, toy_tree, n_markers=3)
    assert (a.obs.loc[ast, "rule_id"] != "G_AS").mean() > 0.5


def test_rule_based_accuracy(toy_adata, toy_tree):
    a = _counts_adata(toy_adata)
    ct.pp.preprocess(a, min_counts=1, knn_smooth_k=None)
    annotate_rule_based(a, toy_tree, n_markers=3)
    assert _accuracy(a.obs["truth"], a.obs["rule_id"]) > 0.85


def test_rule_based_markers_dict(toy_adata, toy_tree):
    a = _counts_adata(toy_adata)
    ct.pp.preprocess(a, min_counts=1, knn_smooth_k=None)
    md = {"excitatory": ["SLC17A7", "SATB2", "CUX2"], "inhibitory": ["GAD1", "GAD2", "SST"]}
    annotate_rule_based(a, toy_tree, markers_dict=md)
    assert set(a.obsm["rule_scores"].columns) == {"excitatory", "inhibitory"}


def test_annotate_rule_based(toy_adata, toy_tree):
    a = ct.annotate(_counts_adata(toy_adata), tree=toy_tree, method="rule_based", n_markers=3, min_cells=10)
    assert "rule_label" in a.obs
    assert a.uns["celltyping"]["method"] == "rule_based"
    assert "knn_smooth" not in a.layers
