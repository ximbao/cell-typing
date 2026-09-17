"""Rule-based prior-knowledge annotation: fixed gene panels, all markers must be detected (count > 0).

Scores are the sum of raw transcript counts across the panel; the highest-scoring qualifying panel wins.
Ties are broken by panel order (tree leaf order, or the order of keys in ``markers_dict``).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..knowledge.tree import CellTypeTree
from ..utils import log
from .flat import candidate_nodes
from .scoring import UNASSIGNED


def _count_matrix(adata: ad.AnnData, genes: list[str]) -> np.ndarray:
    """Cells x genes raw counts (``layers['counts']`` when present, else ``X``)."""
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    idx = [adata.var_names.get_loc(g) for g in genes]
    sub = X[:, idx]
    if sp.issparse(sub):
        return np.asarray(sub.todense(), dtype=float)
    return np.asarray(sub, dtype=float)


def panels_from_tree(tree: CellTypeTree, level: str | int = "leaves", n_markers: int = 4,
                     markers_dict: dict[str, list[str]] | None = None, top_k: int | None = None) -> dict[str, list[str]]:
    """``{node_id: [gene, ...]}`` in competition order. ``markers_dict`` maps label -> genes (looked up on the tree)."""
    if markers_dict is not None:
        out: dict[str, list[str]] = {}
        label2id = {n.label: nid for nid, n in tree.nodes.items() if nid != tree.root}
        for label, genes in markers_dict.items():
            nid = label if label in tree.nodes else label2id.get(label)
            if nid is None:
                log.warning("rule_based: skip unknown label %r (not in tree)", label)
                continue
            out[nid] = list(genes)
        return out
    k = top_k if top_k is not None else n_markers
    nodes = candidate_nodes(tree, level)
    sets = tree.marker_sets(nodes, panel_only=True, top_k=k)
    return {nid: list(gw) for nid, gw in sets.items() if gw}


def annotate_rule_based(adata: ad.AnnData, tree: CellTypeTree, level: str | int = "leaves", n_markers: int = 4,
                        markers_dict: dict[str, list[str]] | None = None, top_k: int | None = None,
                        key: str = "rule") -> ad.AnnData:
    """Panel rule: every gene in the panel must have count > 0; winner = highest sum of raw counts.

    Adds ``obs[key+'_label'|'_id'|'_score']`` and ``obsm[key+'_scores']`` (raw count sums; 0 where the panel did not
    qualify). Uses ``layers['counts']`` when present (after :func:`celltyping.pp.preprocess`), otherwise ``X``.
    """
    panels = panels_from_tree(tree, level, n_markers, markers_dict, top_k)
    panels = {nid: [g for g in genes if g in adata.var_names] for nid, genes in panels.items() if genes}
    if not panels:
        raise ValueError("no marker panels with genes present in adata.var_names")
    node_ids = list(panels.keys())
    labels = [tree.nodes[nid].label for nid in node_ids]
    n_cells, n_panels = adata.n_obs, len(node_ids)

    scores = np.zeros((n_cells, n_panels), dtype=np.float32)
    qualifies = np.zeros((n_cells, n_panels), dtype=bool)
    for j, genes in enumerate(panels.values()):
        x = _count_matrix(adata, genes)
        qualifies[:, j] = (x > 0).all(axis=1)
        scores[:, j] = np.where(qualifies[:, j], x.sum(axis=1), 0.0)

    panel_order = np.arange(n_panels, dtype=float).reshape(1, -1)
    rank = scores + panel_order * 1e-12
    rank = np.where(qualifies, rank, -np.inf)
    best = rank.argmax(axis=1)
    has_hit = qualifies.any(axis=1)

    ids = np.full(n_cells, UNASSIGNED, dtype=object)
    labs = np.full(n_cells, UNASSIGNED, dtype=object)
    sc = np.full(n_cells, np.nan, dtype=np.float32)
    ids[has_hit] = np.array(node_ids, dtype=object)[best[has_hit]]
    labs[has_hit] = np.array(labels, dtype=object)[best[has_hit]]
    sc[has_hit] = scores[has_hit, best[has_hit]]

    adata.obs[f"{key}_id"] = ids
    adata.obs[f"{key}_label"] = pd.Categorical(labs)
    adata.obs[f"{key}_score"] = sc
    score_df = pd.DataFrame(scores, index=adata.obs_names, columns=labels, dtype=np.float32)
    score_df[~qualifies] = 0.0
    adata.obsm[f"{key}_scores"] = score_df
    adata.uns[f"{key}_params"] = {
        "level": str(level), "n_markers": n_markers, "top_k": top_k, "n_panels": n_panels,
        "panels": {tree.nodes[nid].label: panels[nid] for nid in node_ids},
        "rule": "all_genes_gt_zero; score=sum_counts; tie=panel_order",
    }
    log.info("rule_based: %.1f%% unassigned (%d panels, %d genes/panel max)", 100 * (~has_hit).mean(), n_panels, n_markers)
    return adata
