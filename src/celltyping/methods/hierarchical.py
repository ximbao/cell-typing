"""Hierarchical prior-knowledge annotation: walk the ontology tree top-down.

At each internal node, only the cells currently assigned to that node are scored against
that node's children. A cell descends to the best-scoring child if the score passes the
gate (``min_score`` and ``min_margin`` over the runner-up); otherwise it stops and keeps
the (coarser) parent label. Scores are, by default, re-standardised within the cells at
each node so that sub-type markers are contrasted only against related cells.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from ..knowledge.tree import CellTypeTree
from ..utils import log
from .scoring import UNASSIGNED, assign, score_sets, smooth_labels


def annotate_hierarchical(adata: ad.AnnData, tree: CellTypeTree, method: str = "robust_z", min_score: float = 0.5,
                          min_margin: float = 0.25, min_cells: int = 20, rescore_per_node: bool = True,
                          top_k: int | None = 30, subtree_signatures: bool = True, parent_as_competitor: bool = True,
                          layer: str | None = None, key: str = "hier", smooth: dict | None = None,
                          scoring_kwargs: dict | None = None, subtree_mode: str = "union") -> ad.AnnData:
    """Adds per-level labels ``obs[key_level{d}]``, the final ``obs[key_label|key_id|key_depth|key_score|key_margin]``
    and the full score table ``obsm[key_scores]`` (NaN where a node was not evaluated for a cell).

    With ``subtree_signatures`` (default) a child is scored by the markers of its whole subtree,
    made sibling-specific, so that lineage-level decisions (neuron vs glia vs vascular) use the
    well-characterised leaf markers instead of the sparse markers databases list for coarse terms.

    ``subtree_mode`` controls how a child with descendants is scored: ``'union'`` merges the subtree's
    markers into one signature; ``'max'`` scores each informative member of the subtree separately and
    takes the per-cell maximum, which is better for heterogeneous subtrees (a cell matching any one
    member is recognised instead of being diluted by the union).

    With ``parent_as_competitor`` (default) the parent's *own* markers are scored alongside the
    children (except at the root); a cell only descends when the best child beats "stay here" by
    ``min_margin``. This stops over-specific calls such as astrocyte -> Bergmann glia when the
    cell is simply a generic astrocyte.
    """
    sk = dict(scoring_kwargs or {})
    n = adata.n_obs
    current = np.array([tree.root] * n, dtype=object)
    depth_arr = np.zeros(n, dtype=int)
    score_arr = np.full(n, np.nan, dtype=np.float32)
    margin_arr = np.full(n, np.nan, dtype=np.float32)
    non_root = [i for i in tree.nodes if i != tree.root]
    all_scores = pd.DataFrame(np.nan, index=adata.obs_names, columns=non_root, dtype=np.float32)

    def _child_sets(kids: list[str]) -> dict[str, dict[str, float]]:
        if subtree_signatures:
            return tree.subtree_marker_sets(kids, top_k=top_k, sibling_specific=True)
        return tree.marker_sets(kids, top_k=top_k)

    global_scores = None
    if not rescore_per_node and subtree_signatures and subtree_mode == "max":
        log.warning("subtree_mode='max' requires per-node scoring; enabling rescore_per_node")
        rescore_per_node = True
    if not rescore_per_node:
        # one global scoring pass; sets are built per sibling group so specificity is preserved
        all_sets = {}
        for n in tree.iter_bfs():
            if n.children:
                all_sets.update(_child_sets([c for c in n.children if tree.nodes[c].panel_markers or tree.descendants(c)]))
        global_scores = score_sets(adata, all_sets, method=method, layer=layer, **sk)

    for node in tree.iter_bfs():  # parents are processed before children
        if not node.children:
            continue
        mask = current == node.id
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        kids = [c for c in node.children if tree.nodes[c].panel_markers or (subtree_signatures and tree.descendants(c))]
        if not kids:
            continue
        if len(idx) < min_cells:
            log.debug("node %s: only %d cells, not descending", node.label, len(idx))
            continue
        owner: dict[str, str] = {}  # scored set key -> child id (for subtree_mode='max')
        if subtree_signatures and subtree_mode == "max":
            sets = {}
            for kid, members in tree.subtree_member_sets(kids, top_k=top_k, sibling_specific=True).items():
                for member, gw in members.items():
                    if gw:
                        sk_key = kid if member == kid else f"{kid}|{member}"
                        sets[sk_key] = gw
                        owner[sk_key] = kid
        else:
            sets = _child_sets(kids)
            sets = {k: v for k, v in sets.items() if v}
        if not sets:
            continue
        stay_key = f"__stay__{node.id}"
        if parent_as_competitor and node.id != tree.root and node.panel_markers:
            sets[stay_key] = tree.marker_sets([node.id], top_k=top_k)[node.id]
        if rescore_per_node:
            sub = adata[idx]
            scores = score_sets(sub, sets, method=method, layer=layer, **sk)
        else:
            cols = [k for k in sets if k in global_scores.columns]
            scores = global_scores.iloc[idx][cols]
            if stay_key in sets and stay_key not in scores.columns:
                scores[stay_key] = score_sets(adata[idx], {stay_key: sets[stay_key]}, method=method, layer=layer, **sk)[stay_key].to_numpy()
        if owner:  # collapse member sets to their child: per-cell maximum
            agg = {}
            for kid in kids:
                cols = [c for c in scores.columns if owner.get(c) == kid]
                if cols:
                    agg[kid] = scores[cols].max(axis=1)
            if stay_key in scores.columns:
                agg[stay_key] = scores[stay_key]
            scores = pd.DataFrame(agg, index=scores.index)
        child_cols = [c for c in scores.columns if c != stay_key]
        if not child_cols:
            continue
        all_scores.iloc[idx, [all_scores.columns.get_loc(c) for c in child_cols]] = scores[child_cols].to_numpy()
        # single child (and no 'stay' competitor): no runner-up, gate on score only
        res = assign(scores, min_score=min_score, min_margin=(min_margin if scores.shape[1] > 1 else 0.0))
        ok = (res["label"].to_numpy() != UNASSIGNED) & (res["label"].to_numpy() != stay_key)
        sel = idx[ok]
        current[sel] = res["label"].to_numpy()[ok]
        depth_arr[sel] = tree.depth(node.id) + 1
        score_arr[sel] = res["score"].to_numpy()[ok]
        margin_arr[sel] = res["margin"].to_numpy()[ok]
        log.info("node %-40s cells=%7d -> descended %6.1f%% into %d children", node.label[:40], len(idx), 100 * ok.mean(), len(child_cols))

    # per-level columns
    # per-level columns: label of the ancestor at depth d; cells that stopped shallower
    # carry their (coarser) final label down; cells still at root are unassigned.
    max_d = tree.max_depth()
    paths = {nid: tree.path(nid) for nid in tree.nodes}
    labels_final = np.array([tree.nodes[c].label if c != tree.root else UNASSIGNED for c in current], dtype=object)
    for d in range(1, max_d + 1):
        lvl = {}
        for nid, p in paths.items():
            lvl[nid] = tree.nodes[p[d]].label if len(p) > d else (UNASSIGNED if nid == tree.root else tree.nodes[nid].label)
        adata.obs[f"{key}_level{d}"] = pd.Categorical([lvl[c] for c in current])
    adata.obs[f"{key}_id"] = current
    adata.obs[f"{key}_label"] = pd.Categorical(labels_final)
    adata.obs[f"{key}_depth"] = depth_arr
    adata.obs[f"{key}_score"] = score_arr
    adata.obs[f"{key}_margin"] = margin_arr
    adata.obsm[f"{key}_scores"] = all_scores.rename(columns={i: tree.nodes[i].label for i in non_root})
    adata.uns[f"{key}_params"] = {"method": method, "min_score": min_score, "min_margin": min_margin, "min_cells": min_cells,
                                  "rescore_per_node": rescore_per_node, "top_k": top_k, "subtree_signatures": subtree_signatures,
                                  "parent_as_competitor": parent_as_competitor, "subtree_mode": subtree_mode, "layer": layer, "scoring_kwargs": sk,
                                  "tree_meta": tree.meta}
    counts = pd.Series(labels_final).value_counts()
    log.info("hierarchical annotation: %.1f%% unassigned at root; %d distinct labels; top: %s",
             100 * (current == tree.root).mean(), counts.size, ", ".join(f"{k}={v}" for k, v in counts.head(8).items()))
    if smooth:
        smooth_labels(adata, f"{key}_label", conf_key=f"{key}_margin", **smooth)
    return adata


def summarize_levels(adata: ad.AnnData, key: str = "hier") -> pd.DataFrame:
    """Cell counts per label at each level of the hierarchy (long format)."""
    cols = [c for c in adata.obs.columns if c.startswith(f"{key}_level")]
    rows = []
    for c in cols:
        vc = adata.obs[c].value_counts()
        for lab, cnt in vc.items():
            rows.append({"level": int(c.replace(f"{key}_level", "")), "label": lab, "n_cells": int(cnt), "frac": cnt / adata.n_obs})
    return pd.DataFrame(rows)
