"""The one-call entry point: ``celltyping.annotate(adata, tissue="ovary")``."""

from __future__ import annotations

import json
import time
from pathlib import Path

import anndata as ad

from . import pp, tl
from .knowledge.build import Overrides
from .knowledge.tree import CellTypeTree
from .settings import settings
from .utils import log


def annotate(adata: ad.AnnData, tissue: str | None = None, species: str | None = None, tree: CellTypeTree | str | Path | None = None,
             method: str = "hierarchical", overrides: str | Path | Overrides | None = None, preprocess: bool | str = "auto",
             min_counts: int = 20, knn_smooth: int | None = 15, min_score: float = 0.5, min_margin: float = 0.25, top_k: int | None = 30,
             key: str | None = None, knowledge_kwargs: dict | None = None, copy: bool = False, **method_kwargs) -> ad.AnnData:
    """Annotate cells with ontology-derived cell types for a tissue.

    Steps (each skipped when already done):

    1. **knowledge** -- :func:`celltyping.tl.knowledge`: tissue -> expected cell types (Cell Ontology / Uberon,
       marker databases, CZ CELLxGENE Census) -> marker hierarchy restricted to the genes in ``adata``; cached.
    2. **preprocess** -- gene-symbol harmonisation and the QC flag (``obs['qc_flag']``: cells with fewer than
       ``min_counts`` transcripts are ``'low_quality'`` and excluded from everything downstream); if ``X`` holds raw
       counts also ``normalize_total`` + ``log1p`` (counts kept in ``layers['counts']``); then kNN expression smoothing
       over high-quality cells (``layers['knn_smooth']``).
    3. **annotate** -- ``'hierarchical'`` (default; top-down through the tree), ``'flat'`` (all leaves at once)
       or ``'cluster'`` (Leiden + DE + marker overlap).

    Parameters
    ----------
    adata
        Cells x genes; raw counts or log-normalised. Spatial coordinates in ``obsm['spatial']`` are optional.
    tissue
        Tissue name or UBERON id, e.g. ``'ovary'``, ``'cerebral cortex'``. Not needed when ``tree`` is given.
    species
        ``'human'`` or ``'mouse'`` (default :attr:`celltyping.settings.species`).
    tree
        A prebuilt :class:`CellTypeTree` or path to a saved ``*_tree.json``; overrides ``tissue``.
    method
        ``'hierarchical'`` | ``'flat'`` | ``'cluster'``.
    overrides
        YAML with curated tree/marker edits (see ``configs/overrides_*.yaml``).
    preprocess
        ``'auto'`` (normalise only if ``X`` looks like raw counts), ``True`` or ``False``. The QC flag is always added
        when missing.
    min_counts
        QC threshold: cells with ``total_counts < min_counts`` are flagged ``'low_quality'`` and labelled so.
    knn_smooth
        Neighbours for expression smoothing before scoring (``None``/0 disables). Used only when preprocessing runs
        or when ``layers['knn_smooth']`` is absent.
    min_score, min_margin, top_k, **method_kwargs
        Forwarded to :func:`celltyping.tl.hierarchical` / :func:`celltyping.tl.flat` / :func:`celltyping.tl.clusters`.
    key
        ``obs`` prefix for results (default ``'hier'``, ``'flat'`` or ``'cluster'``).
    knowledge_kwargs
        Extra arguments for :func:`celltyping.tl.knowledge` (``min_markers``, ``max_depth``, ``census``, ``sources``, ...).
    copy
        Return a new object instead of modifying ``adata``. No cells are removed (low-quality cells are flagged), so
        ``adata`` is annotated in place; the object is also returned for convenience.

    Returns
    -------
    The annotated AnnData. Labels are in ``obs[key+'_label']`` (CL ids in ``obs[key+'_id']``), scores in
    ``obsm[key+'_scores']``, the run description in ``uns['celltyping']`` and the tree (JSON) in ``uns['celltyping']['tree']``
    (recover it with :func:`celltyping.tl.get_tree`).
    """
    t0 = time.perf_counter()
    if copy:
        adata = adata.copy()
    species = species or settings.species
    if method not in ("hierarchical", "flat", "cluster"):
        raise ValueError("method must be 'hierarchical', 'flat' or 'cluster'")

    # 1. preprocessing (before the tree so that the panel reflects the harmonised gene symbols)
    if preprocess == "auto":
        preprocess = pp.is_raw_counts(adata)
        log.info("X %s raw counts -> preprocessing %s", "looks like" if preprocess else "does not look like", "on" if preprocess else "off")
    if preprocess:
        adata = pp.preprocess(adata, min_counts=min_counts, knn_smooth_k=knn_smooth, random_state=settings.random_state)
    else:
        pp.harmonize_genes(adata)
        if pp.QC_PASS not in adata.obs:
            pp.qc(adata, min_counts=min_counts)
        if knn_smooth and "knn_smooth" not in adata.layers:
            pp.knn_smooth(adata, k=int(knn_smooth), random_state=settings.random_state)

    # 2. knowledge
    if tree is None:
        if tissue is None:
            raise ValueError("give a tissue (e.g. tissue='ovary') or a prebuilt tree")
        tree = tl.knowledge(tissue, panel=adata, species=species, overrides=overrides, **(knowledge_kwargs or {}))
    elif not isinstance(tree, CellTypeTree):
        tree = CellTypeTree.load(tree)
    in_panel = sum(1 for n in tree.nodes.values() if n.panel_markers)
    log.info("tree: %d cell types (%d with panel markers), %d leaves", len(tree) - 1, in_panel, len(tree.leaves()))

    # 3. annotation
    if method == "hierarchical":
        key = key or "hier"
        tl.hierarchical(adata, tree, min_score=min_score, min_margin=min_margin, top_k=top_k, key=key, **method_kwargs)
    elif method == "flat":
        key = key or "flat"
        tl.flat(adata, tree, min_score=min_score, min_margin=min_margin, top_k=top_k, key=key, **method_kwargs)
    else:
        key = key or "cluster"
        mk = {"min_score": 0.2, "min_margin": 0.1, "top_k": 50, **method_kwargs}
        tl.clusters(adata, tree, key=key, **mk)

    label_col = f"{key}_label" if method != "cluster" else next(c for c in adata.obs.columns if c.startswith(f"{key}_r") and c.endswith("_label"))
    counts = adata.obs[label_col].value_counts()
    n_low = int((adata.obs[label_col].astype(str) == pp.LOW_QUALITY).sum())
    adata.uns.setdefault("celltyping", {})
    adata.uns["celltyping"].update({
        "tissue": tree.meta.get("tissue"), "species": species, "method": method, "key": key, "label_column": label_col,
        "tree": json.dumps(tree.to_dict()), "n_types_called": int((counts > 0).sum()), "n_low_quality": n_low,
        "min_counts": min_counts, "runtime_sec": round(time.perf_counter() - t0, 1),
    })
    top = ", ".join(f"{l} {c / adata.n_obs:.0%}" for l, c in counts.head(6).items())
    log.info("annotated %d cells (%d low quality) in %.0fs -> obs['%s'] (%d types; %s)", adata.n_obs, n_low, time.perf_counter() - t0,
             label_col, len(counts), top)
    return adata
