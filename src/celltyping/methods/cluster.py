"""Cluster-based annotation baseline: leiden -> DE genes -> label clusters with the same marker knowledge.

Cluster labelling is automated so that the comparison against the per-cell methods is
reproducible, but the DE tables are kept in ``adata.uns`` for manual curation and a
``apply_manual_labels`` helper accepts a hand-made cluster -> label mapping.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import hypergeom

from ..knowledge.tree import CellTypeTree
from ..preprocess import embed
from ..utils import log
from .flat import candidate_nodes
from .scoring import UNASSIGNED, assign


def run_leiden(adata: ad.AnnData, resolutions: tuple[float, ...] = (0.5, 1.0), key: str = "cluster", random_state: int = 0) -> list[str]:
    if "neighbors" not in adata.uns:
        embed(adata, random_state=random_state)
    keys = []
    for r in resolutions:
        k = f"{key}_r{r:g}"
        sc.tl.leiden(adata, resolution=r, key_added=k, flavor="igraph", n_iterations=2, directed=False, random_state=random_state)
        log.info("leiden r=%g: %d clusters", r, adata.obs[k].nunique())
        keys.append(k)
    return keys


def rank_genes(adata: ad.AnnData, cluster_key: str, n_top: int = 25, method: str = "wilcoxon") -> pd.DataFrame:
    """Wilcoxon DE per cluster; returns a long table (cluster, rank, gene, score, logfc, pval_adj) and stores it in ``uns``."""
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method=method, n_genes=n_top, key_added=f"rgg_{cluster_key}")
    rows = []
    res = adata.uns[f"rgg_{cluster_key}"]
    for cl in res["names"].dtype.names:
        for rank in range(n_top):
            rows.append({"cluster": cl, "rank": rank + 1, "gene": res["names"][cl][rank], "score": float(res["scores"][cl][rank]),
                         "logfc": float(res["logfoldchanges"][cl][rank]), "pval_adj": float(res["pvals_adj"][cl][rank])})
    df = pd.DataFrame(rows)
    adata.uns[f"{cluster_key}_de"] = df
    return df


def cluster_pseudobulk_z(adata: ad.AnnData, cluster_key: str, genes: list[str]) -> pd.DataFrame:
    """Cluster x gene matrix of z-scored mean expression (z across clusters)."""
    X = adata[:, genes].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    groups = adata.obs[cluster_key].astype(str).to_numpy()
    cls = np.unique(groups)
    means = np.vstack([X[groups == c].mean(axis=0) for c in cls])
    mu, sd = means.mean(axis=0), means.std(axis=0)
    sd[sd == 0] = 1.0
    return pd.DataFrame((means - mu) / sd, index=cls, columns=genes)


def label_clusters(adata: ad.AnnData, tree: CellTypeTree, cluster_key: str, level: str | int = "leaves", min_score: float = 0.2,
                   min_margin: float = 0.1, n_top_de: int = 25, top_k: int | None = 50) -> pd.DataFrame:
    """Assign each cluster a cell type using (a) pseudo-bulk marker-set z-scores and (b) DE/marker overlap enrichment.

    The pseudo-bulk score decides the label; the hypergeometric overlap of the cluster's top DE
    genes with the chosen type's markers is reported as supporting evidence.
    """
    nodes = candidate_nodes(tree, level)
    sets = {k: v for k, v in tree.marker_sets(nodes, top_k=top_k).items() if v}
    genes = sorted({g for gw in sets.values() for g in gw} & set(adata.var_names))
    sets = {k: {g: w for g, w in gw.items() if g in genes} for k, gw in sets.items()}
    sets = {k: v for k, v in sets.items() if v}
    Zc = cluster_pseudobulk_z(adata, cluster_key, genes)
    scores = pd.DataFrame(index=Zc.index)
    for k, gw in sets.items():
        w = pd.Series(gw)
        w = w / w.sum()
        scores[k] = Zc[w.index].to_numpy() @ w.to_numpy()
    res = assign(scores, min_score=min_score, min_margin=min_margin)
    de = adata.uns.get(f"{cluster_key}_de")
    if de is None:
        de = rank_genes(adata, cluster_key, n_top=n_top_de)
    N = adata.n_vars
    rows = []
    for cl, r in res.iterrows():
        top = set(de.loc[de["cluster"] == cl, "gene"].head(n_top_de))
        if r["label"] != UNASSIGNED:
            mk = set(sets[r["label"]])
            ov = len(top & mk)
            p = hypergeom.sf(ov - 1, N, len(mk), len(top))
        else:
            ov, p, mk = 0, np.nan, set()
        rows.append({"cluster": cl, "n_cells": int((adata.obs[cluster_key].astype(str) == cl).sum()),
                     "label_id": r["label"], "label": tree.nodes[r["label"]].label if r["label"] in tree else UNASSIGNED,
                     "score": r["score"], "margin": r["margin"],
                     "runner_up": tree.nodes[r["runner_up"]].label if r["runner_up"] in tree else None,
                     "de_marker_overlap": ov, "overlap_pval": p,
                     "shared_de_markers": ", ".join(sorted(top & mk)),
                     "top_de": ", ".join(de.loc[de["cluster"] == cl, "gene"].head(8))})
    table = pd.DataFrame(rows).set_index("cluster")
    adata.uns[f"{cluster_key}_labels"] = table.reset_index()
    adata.obsm[f"{cluster_key}_type_scores"] = scores.rename(columns={i: tree.nodes[i].label for i in scores.columns}).reindex(
        adata.obs[cluster_key].astype(str)).set_index(adata.obs_names)
    mapping = table["label"].to_dict()
    mapping_id = table["label_id"].to_dict()
    adata.obs[f"{cluster_key}_label"] = pd.Categorical(adata.obs[cluster_key].astype(str).map(mapping))
    adata.obs[f"{cluster_key}_id"] = adata.obs[cluster_key].astype(str).map(mapping_id).to_numpy()
    n_un = int((table["label_id"].astype(str) == UNASSIGNED).to_numpy().sum())
    log.info("%s: labelled %d clusters (%d unassigned)", cluster_key, len(table), n_un)
    return table


def apply_manual_labels(adata: ad.AnnData, cluster_key: str, mapping: dict[str, str], out_key: str | None = None) -> pd.Series:
    """Map cluster ids to hand-curated labels (unlisted clusters become ``unassigned``)."""
    out_key = out_key or f"{cluster_key}_manual"
    adata.obs[out_key] = pd.Categorical(adata.obs[cluster_key].astype(str).map(lambda c: mapping.get(str(c), UNASSIGNED)))
    return adata.obs[out_key]


def annotate_clusters(adata: ad.AnnData, tree: CellTypeTree, resolutions: tuple[float, ...] = (0.5, 1.0), level: str | int = "leaves",
                      min_score: float = 0.2, min_margin: float = 0.1, n_top_de: int = 25, top_k: int | None = 50,
                      key: str = "cluster", random_state: int = 0) -> dict[str, pd.DataFrame]:
    """Full baseline: leiden at each resolution, DE genes, automated cluster labels. Returns ``{cluster_key: label_table}``."""
    keys = run_leiden(adata, resolutions, key=key, random_state=random_state)
    out = {}
    for k in keys:
        rank_genes(adata, k, n_top=n_top_de)
        out[k] = label_clusters(adata, tree, k, level=level, min_score=min_score, min_margin=min_margin, n_top_de=n_top_de, top_k=top_k)
    adata.uns[f"{key}_params"] = {"resolutions": list(resolutions), "level": str(level), "min_score": min_score, "min_margin": min_margin,
                                  "n_top_de": n_top_de, "top_k": top_k}
    return out
