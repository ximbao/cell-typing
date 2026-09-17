"""Shared QC and normalisation so every annotation method sees identical input."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .utils import log, norm_gene


QC_PASS = "qc_pass"
QC_FLAG = "qc_flag"
HIGH_QUALITY = "high_quality"
LOW_QUALITY = "low_quality"


def qc_filter(adata: ad.AnnData, min_counts: int = 20, min_genes: int = 0, max_control_frac: float = 0.1,
              min_cell_area: float | None = None, max_cell_area: float | None = None, filter: bool = False) -> ad.AnnData:
    """Per-cell QC for imaging-based ST: **flags** cells instead of removing them.

    Adds ``obs['qc_pass']`` (bool) and ``obs['qc_flag']`` (``'high_quality'`` / ``'low_quality'``). A cell is low
    quality when ``total_counts < min_counts`` (default 20 transcripts), when it has fewer than ``min_genes`` detected
    genes, when control probes make up more than ``max_control_frac`` of its counts, or when its area is outside
    ``[min_cell_area, max_cell_area]``. Downstream steps (kNN smoothing, all annotators, the benchmark) use only
    high-quality cells; low-quality cells receive the label ``'low_quality'``. With ``filter=True`` the low-quality
    cells are removed instead and a subset is returned.

    Counts are taken from ``layers['counts']`` when ``X`` is already normalised, so the flag can be (re)computed at
    any point of the workflow.
    """
    n0 = adata.n_obs
    normalised = "normalized" in adata.uns or "log1p" in adata.uns
    if "counts" in adata.layers:
        src = adata.layers["counts"]
    elif normalised and "total_counts" in adata.obs:
        src = None  # no counts matrix: rely on the existing per-cell totals
        log.info("QC: X is normalised and no counts layer is present; using obs['total_counts']")
    else:
        if normalised:
            log.warning("QC: X looks normalised and neither layers['counts'] nor obs['total_counts'] exist; totals are computed from X")
        src = adata.X
    if src is not None:
        total = np.asarray(src.sum(axis=1)).ravel()
        n_genes = np.asarray((src > 0).sum(axis=1)).ravel()
        adata.obs["total_counts"] = total
        adata.obs["n_genes_by_counts"] = n_genes
    else:
        total = adata.obs["total_counts"].to_numpy(dtype=float)
        n_genes = adata.obs["n_genes_by_counts"].to_numpy() if "n_genes_by_counts" in adata.obs else np.asarray((adata.X > 0).sum(axis=1)).ravel()
    keep = (total >= min_counts) & (n_genes >= min_genes)
    reasons = {"low_counts": int((total < min_counts).sum()), "few_genes": int((n_genes < min_genes).sum())}
    if "control_counts" in adata.obs:
        frac = adata.obs["control_counts"].to_numpy() / np.clip(total + adata.obs["control_counts"].to_numpy(), 1, None)
        adata.obs["control_frac"] = frac
        keep &= frac <= max_control_frac
        reasons["control_probes"] = int((frac > max_control_frac).sum())
    if min_cell_area is not None and "cell_area" in adata.obs:
        keep &= adata.obs["cell_area"].to_numpy() >= min_cell_area
    if max_cell_area is not None and "cell_area" in adata.obs:
        keep &= adata.obs["cell_area"].to_numpy() <= max_cell_area
    adata.obs[QC_PASS] = keep
    adata.obs[QC_FLAG] = pd.Categorical(np.where(keep, HIGH_QUALITY, LOW_QUALITY), categories=[HIGH_QUALITY, LOW_QUALITY])
    adata.uns["qc"] = {"min_counts": min_counts, "min_genes": min_genes, "max_control_frac": max_control_frac,
                       "n_cells": int(n0), "n_low_quality": int((~keep).sum()), "reasons": reasons}
    log.info("QC: %d / %d cells high quality (%d flagged low quality: %s)", int(keep.sum()), n0, int((~keep).sum()),
             ", ".join(f"{k}={v}" for k, v in reasons.items() if v))
    if filter:
        return adata[keep].copy()
    return adata


def qc_mask(adata: ad.AnnData) -> np.ndarray | None:
    """Boolean mask of high-quality cells, or ``None`` when no QC flag is present / all cells pass."""
    if QC_PASS not in adata.obs:
        return None
    m = adata.obs[QC_PASS].to_numpy().astype(bool)
    return None if m.all() else m


def normalize(adata: ad.AnnData, target_sum: float | None = None) -> ad.AnnData:
    """Store counts in ``layers['counts']``, then normalize_total (median by default) + log1p."""
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.uns["normalized"] = {"target_sum": target_sum, "log1p": True}
    return adata


def harmonize_var_names(adata: ad.AnnData) -> ad.AnnData:
    """Upper-case gene symbols (matching the knowledge tables) while keeping the originals."""
    if "original_symbol" not in adata.var:
        adata.var["original_symbol"] = adata.var_names
    adata.var_names = [norm_gene(g) for g in adata.var_names]
    adata.var_names_make_unique()
    return adata


def preprocess(adata: ad.AnnData, min_counts: int = 20, min_genes: int = 0, max_control_frac: float = 0.1, target_sum: float | None = None) -> ad.AnnData:
    adata = harmonize_var_names(adata)
    adata = qc_filter(adata, min_counts=min_counts, min_genes=min_genes, max_control_frac=max_control_frac)
    adata = normalize(adata, target_sum=target_sum)
    return adata


def embed(adata: ad.AnnData, n_pcs: int = 30, n_neighbors: int = 15, random_state: int = 0, use_gpu: bool = False) -> ad.AnnData:
    """PCA + kNN graph (+ UMAP) used by the clustering baseline and for plotting."""
    if "X_pca" not in adata.obsm:
        # PCA on log-normalised values (implicitly centred); X is left untouched so
        # marker scoring downstream still sees log-normalised expression.
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1), random_state=random_state)
    if use_gpu:
        try:
            import rapids_singlecell as rsc

            rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
            return adata
        except Exception as e:  # pragma: no cover
            log.warning("rapids_singlecell unavailable (%s); falling back to CPU", e)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata.obsm["X_pca"].shape[1]), random_state=random_state)
    return adata


def knn_smooth(adata: ad.AnnData, k: int = 15, layer: str = "knn_smooth", n_pcs: int = 30, random_state: int = 0) -> ad.AnnData:
    """Average each cell's log-normalised expression with its k nearest neighbours in PCA space.

    Imaging-based ST counts are very sparse (a cell often shows 2-4 of its 30 marker genes);
    smoothing over the expression graph denoises per-cell signature scores considerably.
    ``adata.X`` is left untouched (used for clustering/DE); the result goes to ``layers[layer]``.
    """
    import scipy.sparse as sp

    mask = qc_mask(adata)
    if mask is not None:
        # neighbours are searched among high-quality cells only; low-quality cells keep their own (unsmoothed) values
        sub = adata[mask].copy()
        embed(sub, n_pcs=n_pcs, n_neighbors=k, random_state=random_state)
        C = sub.obsp["connectivities"].tocsr(copy=True)
        C.setdiag(1.0)
        C = (sp.diags(1.0 / np.asarray(C.sum(axis=1)).ravel()) @ C).tocoo()
        # embed the (m x m) smoothing operator into an (n x n) one: identity rows for low-quality cells
        idx = np.where(mask)[0]
        low = np.where(~mask)[0]
        C_full = sp.coo_matrix((np.concatenate([C.data, np.ones(len(low))]),
                                (np.concatenate([idx[C.row], low]), np.concatenate([idx[C.col], low]))),
                               shape=(adata.n_obs, adata.n_obs)).tocsr()
        adata.layers[layer] = (C_full @ adata.X).astype(np.float32)
        adata.obsm["X_pca"] = np.full((adata.n_obs, sub.obsm["X_pca"].shape[1]), np.nan, dtype=np.float32)
        adata.obsm["X_pca"][mask] = sub.obsm["X_pca"]
        adata.uns["knn_smooth"] = {"k": k, "layer": layer, "n_cells_smoothed": int(mask.sum())}
        log.info("kNN-smoothed expression (k=%d, %d high-quality cells) stored in layers['%s']", k, int(mask.sum()), layer)
        return adata
    if "neighbors" not in adata.uns or adata.uns["neighbors"]["params"].get("n_neighbors") != k:
        embed(adata, n_pcs=n_pcs, n_neighbors=k, random_state=random_state)
    C = adata.obsp["connectivities"].tocsr(copy=True)
    C.setdiag(1.0)
    C = sp.diags(1.0 / np.asarray(C.sum(axis=1)).ravel()) @ C
    adata.layers[layer] = (C @ adata.X).astype(np.float32)
    adata.uns["knn_smooth"] = {"k": k, "layer": layer}
    log.info("kNN-smoothed expression (k=%d) stored in layers['%s']", k, layer)
    return adata


def gene_sd_floor(adata: ad.AnnData, layer: str | None = None, n_genes: int = 1000, random_state: int = 0) -> float:
    """Regularisation constant ``s0`` for z-scores: the median per-gene SD over (a sample of) all genes.

    Cached in ``adata.uns['zscore_s0']`` (not on views).
    """
    key = f"zscore_s0:{layer or 'X'}"
    if key in adata.uns:
        return float(adata.uns[key])
    rng = np.random.default_rng(random_state)
    genes = adata.var_names if adata.n_vars <= n_genes else rng.choice(adata.var_names, n_genes, replace=False)
    sub = adata[:, list(genes)]
    X = sub.layers[layer] if layer else sub.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    s0 = float(np.median(X.std(axis=0)))
    if not adata.is_view:
        adata.uns[key] = s0
    return s0


def zscore_matrix(adata: ad.AnnData, genes: list[str], clip: float = 10.0, layer: str | None = None, s0: float | None = None,
                  scale: str = "z") -> np.ndarray:
    """Dense cells x genes matrix of per-gene standardised expression (statistics over the cells in ``adata``).

    ``scale='z'``      (x - mean) / sd. With ``s0`` the SD is regularised as ``sqrt(sd^2 + s0^2)`` (SAM-style
                       fudge factor): a gene detected in <1% of cells has a tiny SD, so a single count would
                       otherwise give a huge z and marker sets of rare genes act like lottery tickets.
    ``scale='center'`` x - mean, in log-expression units. Genes are *not* rescaled, so a marker's contribution
                       is proportional to how much it is actually expressed above background; in sparse
                       imaging data this is markedly more robust than per-gene z-scores.
    """
    sub = adata[:, genes]
    X = sub.layers[layer] if layer else sub.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    mu = X.mean(axis=0)
    if scale == "center":
        return np.clip(X - mu, -clip, clip).astype(np.float32)
    if scale != "z":
        raise ValueError(f"unknown scale {scale!r} (use 'z' or 'center')")
    sd = X.std(axis=0)
    if s0:
        sd = np.sqrt(sd**2 + s0**2)
    sd[sd == 0] = 1.0
    return np.clip((X - mu) / sd, -clip, clip).astype(np.float32)
