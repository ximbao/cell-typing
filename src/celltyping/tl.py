"""``celltyping.tl``: tools -- knowledge building and the three annotators.

Typical use::

    tree = ct.tl.knowledge("ovary", panel=adata)          # cached per (tissue, species, panel, overrides, params)
    ct.tl.hierarchical(adata, tree)                        # obs['hier_label'], obs['hier_level1'..], obsm['hier_scores']
    ct.tl.flat(adata, tree)                                # obs['flat_label'], obsm['flat_scores']
    ct.tl.clusters(adata, tree, resolutions=(0.5, 1.0))    # obs['cluster_r0.5_label'], uns['cluster_r0.5_de']
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .knowledge.build import BuildParams, Overrides, build_knowledge
from .knowledge.ontology import expected_cell_types as _expected
from .knowledge.ontology import resolve_tissue
from .knowledge.panel import load_panel
from .knowledge.tree import CellTypeTree
from .methods.cluster import annotate_clusters as _clusters
from .methods.cluster import apply_manual_labels  # noqa: F401
from .methods.flat import annotate_flat as _flat
from .methods.hierarchical import annotate_hierarchical as _hier
from .methods.hierarchical import summarize_levels  # noqa: F401
from .methods.scoring import score_sets, smooth_labels  # noqa: F401
from .preprocess import LOW_QUALITY, qc_mask
from .settings import settings
from .utils import log


# --------------------------------------------------------------------------- knowledge
def _panel_key(panel) -> tuple[list[str] | None, str]:
    if panel is None:
        return None, "nopanel"
    genes = load_panel(panel)
    return genes, hashlib.sha1("\n".join(genes).encode()).hexdigest()[:10]


def _overrides_key(overrides) -> str:
    if overrides is None:
        return "noov"
    if isinstance(overrides, Overrides):
        blob = json.dumps(overrides.__dict__, sort_keys=True, default=str)
    else:
        blob = Path(overrides).read_text()
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def knowledge(tissue: str, panel=None, species: str | None = None, overrides: str | Path | Overrides | None = None,
              sources: tuple[str, ...] | None = None, census: bool | None = None, min_markers: int = 5, max_depth: int = 5,
              ancestor_levels: int = 1, min_db_sources: int = 2, cache: bool = True, force: bool = False,
              out_dir: str | Path | None = None, **build_params) -> CellTypeTree:
    """Build (or load from cache) the tissue cell-type tree with panel-restricted markers.

    Parameters
    ----------
    tissue
        Tissue name or UBERON id (``'ovary'``, ``'cerebral cortex'``, ``'UBERON:0002048'``).
    panel
        Gene panel: an AnnData (its ``var_names``), a Xenium ``gene_panel.json``, a gene-list file or a list.
        ``None`` keeps all markers (not recommended for scoring).
    species
        ``'human'`` or ``'mouse'`` (default :attr:`celltyping.settings.species`).
    overrides
        YAML path or :class:`Overrides` with curated additions/removals/marker edits.
    sources, census
        Marker sources and whether to use CZ CELLxGENE Census observations (defaults from settings).
    min_markers, max_depth, ancestor_levels, min_db_sources, **build_params
        Forwarded to :class:`celltyping.knowledge.build.BuildParams`.
    cache
        Cache the built tree under ``settings.knowledge_dir`` keyed by tissue, species, panel, overrides and params.
    out_dir
        Additionally write ``<slug>_tree.json/.txt`` and the coverage/dropped tables there.
    """
    species = species or settings.species
    params = BuildParams(species=species, sources=tuple(sources or settings.sources), ancestor_levels=ancestor_levels,
                         min_db_sources=min_db_sources, min_markers=min_markers, max_depth=max_depth,
                         census=settings.census if census is None else census,
                         **({"census_version": settings.census_version} if settings.census_version else {}), **build_params)
    genes, pkey = _panel_key(panel)
    t = resolve_tissue(tissue)
    pblob = json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in params.__dict__.items()}, sort_keys=True)
    key = hashlib.sha1(f"{t.id}|{pkey}|{_overrides_key(overrides)}|{pblob}".encode()).hexdigest()[:12]
    cache_file = settings.knowledge_dir / f"{t.slug}_{key}_tree.json"
    if cache and cache_file.exists() and not force:
        log.info("knowledge: loading cached tree for %s (%s)", t.label, cache_file.name)
        tree = CellTypeTree.load(cache_file)
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            tree.save(Path(out_dir) / f"{t.slug}_tree.json")
            (Path(out_dir) / f"{t.slug}_tree.txt").write_text(tree.render())
        return tree
    tree, cov, dropped = build_knowledge(t.id, panel=genes, params=params, overrides=overrides, out_dir=out_dir, force=force)
    if cache:
        tree.save(cache_file)
        cov.to_csv(cache_file.with_name(f"{t.slug}_{key}_coverage.csv"), index=False)
        dropped.to_csv(cache_file.with_name(f"{t.slug}_{key}_dropped.csv"), index=False)
    return tree


def expected_cell_types(tissue: str, species: str | None = None, ancestor_levels: int = 1, census: bool | None = None,
                        census_disease: str = "normal", census_min_cells: int = 200, census_min_datasets: int = 2,
                        census_min_frac: float = 0.001) -> pd.DataFrame:
    """Cell types the ontology (and optionally the CELLxGENE Census) place in a tissue, before marker filtering.

    Columns: ``cl_id, label, relation, tissue_id, tissue_label, census_n_cells, census_n_datasets`` (Census counts for
    healthy samples by default; rows with ``relation == '-'`` are Census-only observations).
    """
    species = species or settings.species
    t = resolve_tissue(tissue)
    df = _expected(t, species, ancestor_levels)
    df["census_n_cells"] = pd.NA
    df["census_n_datasets"] = pd.NA
    if settings.census if census is None else census:
        from .knowledge.cellxgene import census_cell_types, census_seeds, census_tissue_general

        tg = census_tissue_general(t)
        if tg is not None:
            cx = census_seeds(census_cell_types(tg[1], species, disease=census_disease), species, census_min_cells,
                              census_min_datasets, census_min_frac)
            cx = cx.rename(columns={"n_cells": "census_n_cells", "n_datasets": "census_n_datasets"})
            extra = cx[~cx["cl_id"].isin(df["cl_id"])].assign(relation="-", tissue_id=tg[0], tissue_label=f"Census:{tg[1]}")
            df = df.drop(columns=["census_n_cells", "census_n_datasets"]).merge(
                cx[["cl_id", "census_n_cells", "census_n_datasets"]], on="cl_id", how="left")
            df = pd.concat([df, extra[df.columns]], ignore_index=True)
    return df.sort_values("label").reset_index(drop=True)


def get_tree(adata: ad.AnnData) -> CellTypeTree:
    """The tree stored by :func:`celltyping.annotate` in ``uns['celltyping']['tree']``."""
    return CellTypeTree.from_dict(json.loads(adata.uns["celltyping"]["tree"]))


# --------------------------------------------------------------------------- annotators
def _on_high_quality(adata: ad.AnnData, fn, label_suffixes=("_label", "_id", "_level")):
    """Run ``fn(sub)`` on the high-quality cells only and write its results back into ``adata``.

    New ``obs`` columns are transferred with ``'low_quality'`` in label/id columns and NaN in numeric ones; new
    ``obsm`` tables are re-indexed to all cells (NaN rows); new ``uns`` entries are copied. Returns ``fn``'s result.
    """
    mask = qc_mask(adata)
    if mask is None:
        return fn(adata)
    sub = adata[mask].copy()
    before_obs, before_obsm, before_uns = set(adata.obs.columns), set(adata.obsm.keys()), set(adata.uns.keys())
    out = fn(sub)
    for c in [c for c in sub.obs.columns if c not in before_obs]:
        col = sub.obs[c]
        if any(suf in c for suf in label_suffixes) and not pd.api.types.is_numeric_dtype(col):
            full = pd.Series(LOW_QUALITY, index=adata.obs_names, dtype=object)
            full[sub.obs_names] = col.astype(str).to_numpy()
            adata.obs[c] = pd.Categorical(full) if isinstance(col.dtype, pd.CategoricalDtype) else full
        else:
            full = pd.Series(np.nan, index=adata.obs_names, dtype=float) if pd.api.types.is_numeric_dtype(col) \
                else pd.Series(None, index=adata.obs_names, dtype=object)
            full[sub.obs_names] = col.to_numpy()
            adata.obs[c] = full
    for k in [k for k in sub.obsm.keys() if k not in before_obsm]:
        v = sub.obsm[k]
        if isinstance(v, pd.DataFrame):
            adata.obsm[k] = v.reindex(adata.obs_names)
        else:
            arr = np.full((adata.n_obs, v.shape[1]), np.nan, dtype=np.float32)
            arr[mask] = v
            adata.obsm[k] = arr
    for k in [k for k in sub.uns.keys() if k not in before_uns]:
        adata.uns[k] = sub.uns[k]
    log.info("%d low-quality cells labelled '%s'", int((~mask).sum()), LOW_QUALITY)
    return out


def _layer(adata: ad.AnnData, layer: str | None | bool) -> str | None:
    """``layer='auto'`` -> ``'knn_smooth'`` if present."""
    if layer == "auto" or layer is True:
        return "knn_smooth" if "knn_smooth" in adata.layers else None
    return None if layer is False else layer


def hierarchical(adata: ad.AnnData, tree: CellTypeTree, method: str = "robust_z", min_score: float = 0.5, min_margin: float = 0.25,
                 min_cells: int = 20, top_k: int | None = 30, subtree_signatures: bool = True, subtree_mode: str = "union",
                 parent_as_competitor: bool = True, rescore_per_node: bool = True, layer: str | None = "auto", key: str = "hier",
                 smooth: dict | None = None, scale: str = "center", top_frac: float = 0.5, **scoring_kwargs) -> ad.AnnData:
    """Top-down annotation through the tree: at every node, score the children on the cells that reached it and
    descend only when the best child beats the runner-up (and, with ``parent_as_competitor``, the parent) by
    ``min_margin`` with score >= ``min_score``; otherwise the cell stops at the current (coarser) label.

    Results: ``obs[key+'_label'|'_id'|'_depth'|'_score'|'_margin']``, per-level labels ``obs[key+'_level{d}']``,
    ``obsm[key+'_scores']`` and ``uns[key+'_params']``.
    """
    sk = {"scale": scale, "top_frac": top_frac, **scoring_kwargs}
    _on_high_quality(adata, lambda a: _hier(a, tree, method=method, min_score=min_score, min_margin=min_margin, min_cells=min_cells,
                                            rescore_per_node=rescore_per_node, top_k=top_k, subtree_signatures=subtree_signatures,
                                            parent_as_competitor=parent_as_competitor, layer=_layer(a, layer), key=key, smooth=smooth,
                                            scoring_kwargs=sk, subtree_mode=subtree_mode))
    return adata


def flat(adata: ad.AnnData, tree: CellTypeTree, level: str | int = "leaves", method: str = "robust_z", min_score: float = 0.5,
         min_margin: float = 0.25, top_k: int | None = 30, layer: str | None = "auto", key: str = "flat", scale: str = "center",
         top_frac: float = 0.5, **scoring_kwargs) -> ad.AnnData:
    """One-shot signature scoring of all candidates at ``level`` (``'leaves'``, an integer depth or ``'all'``);
    the classical prior-knowledge baseline. Results: ``obs[key+'_label'|'_score'|'_margin']``, ``obsm[key+'_scores']``."""
    sk = {"scale": scale, "top_frac": top_frac, **scoring_kwargs}
    _on_high_quality(adata, lambda a: _flat(a, tree, level=level, method=method, min_score=min_score, min_margin=min_margin,
                                            top_k=top_k, layer=_layer(a, layer), key=key, scoring_kwargs=sk))
    return adata


def clusters(adata: ad.AnnData, tree: CellTypeTree, resolutions: tuple[float, ...] = (0.5, 1.0), level: str | int = "leaves",
             min_score: float = 0.2, min_margin: float = 0.1, n_top_de: int = 25, top_k: int | None = 50, key: str = "cluster",
             random_state: int | None = None) -> dict[str, pd.DataFrame]:
    """Clustering baseline: Leiden at each resolution, Wilcoxon DE per cluster, then clusters are labelled by
    overlap of their DE genes with the tree's markers (``obs[key+'_r{res}_label']``, ``uns[key+'_r{res}_de']``).
    Returns the per-resolution cluster->label tables; edit them and apply with :func:`apply_manual_labels`."""
    return _on_high_quality(adata, lambda a: _clusters(a, tree, resolutions=tuple(resolutions), level=level, min_score=min_score,
                                                       min_margin=min_margin, n_top_de=n_top_de, top_k=top_k, key=key,
                                                       random_state=settings.random_state if random_state is None else random_state))


def score(adata: ad.AnnData, tree: CellTypeTree, nodes: list[str] | None = None, method: str = "robust_z", top_k: int | None = 30,
          layer: str | None = "auto", key: str = "scores", scale: str = "center", top_frac: float = 0.5, **scoring_kwargs) -> pd.DataFrame:
    """Signature scores for arbitrary tree nodes (default: all leaves), stored in ``obsm[key]`` and returned."""
    nodes = nodes or tree.leaves()
    sets = {k: v for k, v in tree.marker_sets(nodes, panel_only=True, top_k=top_k).items() if v}

    def _run(a):
        df = score_sets(a, sets, method=method, layer=_layer(a, layer), scale=scale, top_frac=top_frac, **scoring_kwargs)
        df.columns = [tree.nodes[c].label if c in tree.nodes else c for c in df.columns]
        a.obsm[key] = df

    _on_high_quality(adata, _run)
    return adata.obsm[key]


__all__ = ["knowledge", "get_tree", "expected_cell_types", "hierarchical", "flat", "clusters", "score", "summarize_levels", "smooth_labels",
           "apply_manual_labels"]
