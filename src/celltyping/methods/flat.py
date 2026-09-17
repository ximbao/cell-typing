"""Flat (non-hierarchical) prior-knowledge annotation: score every candidate type at once, take the argmax."""

from __future__ import annotations

import anndata as ad
import pandas as pd

from ..knowledge.tree import CellTypeTree
from ..utils import log
from .scoring import UNASSIGNED, assign, score_sets


def candidate_nodes(tree: CellTypeTree, level: str | int = "leaves") -> list[str]:
    """``'leaves'`` (default), ``'all'`` non-root nodes, or an integer depth (nodes at that depth
    plus shallower leaves so that every cell has a candidate along each branch)."""
    if level == "leaves":
        return tree.leaves()
    if level == "all":
        return [i for i in tree.nodes if i != tree.root]
    d = int(level)
    return [i for i, n in tree.nodes.items() if i != tree.root and (tree.depth(i) == d or (tree.depth(i) < d and not n.children))]


def annotate_flat(adata: ad.AnnData, tree: CellTypeTree, level: str | int = "leaves", method: str = "robust_z",
                  min_score: float = 0.5, min_margin: float = 0.25, top_k: int | None = 30, layer: str | None = None,
                  key: str = "flat", scoring_kwargs: dict | None = None) -> ad.AnnData:
    """Adds ``obs[key_label|key_id|key_score|key_margin]`` and ``obsm[key_scores]``.

    Candidates at an integer ``level`` (or ``'all'``) use subtree signatures for internal nodes.
    """
    nodes = candidate_nodes(tree, level)
    if level == "leaves":
        sets = tree.marker_sets(nodes, panel_only=True, top_k=top_k)
    else:
        sets = tree.subtree_marker_sets(nodes, panel_only=True, top_k=top_k, sibling_specific=True)
    sets = {k: v for k, v in sets.items() if v}
    log.info("flat annotation: %d candidate types (%s), method=%s", len(sets), level, method)
    scores = score_sets(adata, sets, method=method, layer=layer, **(scoring_kwargs or {}))
    res = assign(scores, min_score=min_score, min_margin=min_margin)
    id2label = {i: tree.nodes[i].label for i in sets}
    adata.obs[f"{key}_id"] = res["label"].values
    adata.obs[f"{key}_label"] = pd.Categorical([id2label.get(i, UNASSIGNED) for i in res["label"]])
    adata.obs[f"{key}_score"] = res["score"].values
    adata.obs[f"{key}_margin"] = res["margin"].values
    adata.obsm[f"{key}_scores"] = scores.rename(columns=id2label)
    adata.uns[f"{key}_params"] = {"level": str(level), "method": method, "min_score": min_score, "min_margin": min_margin, "top_k": top_k,
                                  "layer": layer, "scoring_kwargs": dict(scoring_kwargs or {}), "n_candidates": len(sets)}
    frac_un = (res["label"] == UNASSIGNED).mean()
    log.info("flat annotation: %.1f%% unassigned", 100 * frac_un)
    return adata
