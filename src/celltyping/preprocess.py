"""Shared QC and normalisation so every annotation method sees identical input."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .utils import log, norm_gene


QC_PASS = "qc_pass"
QC_FLAG = "qc_flag"
QC_REASON = "qc_reason"
HIGH_QUALITY = "high_quality"
LOW_QUALITY = "low_quality"
# Xenium cells.parquet / h5ad metadata: only true negative controls count towards the fraction threshold.
# Unassigned / deprecated codewords are common at 10-15% and are not a per-cell QC failure on their own.
XENIUM_NEG_CTRL = ("control_probe_counts", "genomic_control_counts")


def _matrix_gene_counts(adata: ad.AnnData) -> np.ndarray:
    """Per-cell gene totals from ``layers['counts']`` or ``X`` (the expression matrix, genes only)."""
    src = adata.layers["counts"] if "counts" in adata.layers else adata.X
    return np.asarray(src.sum(axis=1)).ravel()


def _ensure_transcript_counts(adata: ad.AnnData, matrix_counts: np.ndarray | None = None) -> np.ndarray:
    """Return per-cell transcript counts for QC; fill ``obs['transcript_counts']`` from the matrix when missing."""
    if "transcript_counts" in adata.obs:
        return adata.obs["transcript_counts"].to_numpy(dtype=float)
    counts = matrix_counts if matrix_counts is not None else _matrix_gene_counts(adata)
    adata.obs["transcript_counts"] = counts
    return counts


def _qc_control_counts(adata: ad.AnnData) -> np.ndarray | None:
    """Per-cell negative-control counts for the QC fraction (``None`` -> skip the control check)."""
    if all(c in adata.obs for c in XENIUM_NEG_CTRL):
        return (adata.obs["control_probe_counts"].to_numpy(dtype=float)
                + adata.obs["genomic_control_counts"].to_numpy(dtype=float))
    if "control_counts" in adata.obs:
        return adata.obs["control_counts"].to_numpy(dtype=float)
    return None


def qc_filter(adata: ad.AnnData, min_counts: int = 20, min_genes: int = 0, max_control_frac: float = 0.3,
              min_cell_area: float | None = None, max_cell_area: float | None = None, filter: bool = False) -> ad.AnnData:
    """Per-cell QC for imaging-based ST: **flags** cells instead of removing them.

    Adds ``obs['qc_pass']`` (bool), ``obs['qc_flag']`` (``'high_quality'`` / ``'low_quality'``) and ``obs['qc_reason']``
    (``'pass'``, ``'low_counts'``, ``'few_genes'``, ``'high_control_frac'``, ``'cell_area'``). A cell is low quality
    when ``transcript_counts < min_counts`` (default 20 gene transcripts per cell), when it has fewer than ``min_genes``
    detected genes, when **negative** control probes make up more than ``max_control_frac`` of transcript + control
    counts (on Xenium data only ``control_probe_counts`` + ``genomic_control_counts`` are used -- not
    unassigned/deprecated codewords, which are often 10-15% and do not mean the cell is bad), or when its area is
    outside ``[min_cell_area, max_cell_area]``. Downstream steps use only high-quality cells; low-quality cells receive
    the label ``'low_quality'``. With ``filter=True`` the low-quality cells are removed instead.

    ``obs['transcript_counts']`` is the column used for ``min_counts`` (from Xenium metadata when present, otherwise
    the sum of ``X`` / ``layers['counts']``). Xenium's ``total_counts`` metadata (gene + control codewords) is never
    used for this threshold and is left unchanged when already present.
    """
    n0 = adata.n_obs
    normalised = "normalized" in adata.uns or "log1p" in adata.uns
    if "counts" in adata.layers:
        src = adata.layers["counts"]
    elif not normalised:
        src = adata.X
    else:
        src = None
        if "transcript_counts" not in adata.obs:
            log.warning("QC: X is normalised and neither layers['counts'] nor obs['transcript_counts'] exist; "
                        "transcript totals are computed from X (unreliable on log-normalised data)")
            src = adata.X
        else:
            log.info("QC: using obs['transcript_counts'] for min_counts")
    if src is not None:
        matrix_counts = np.asarray(src.sum(axis=1)).ravel()
        n_genes = np.asarray((src > 0).sum(axis=1)).ravel()
        adata.obs["n_genes_by_counts"] = n_genes
        if "total_counts" not in adata.obs:
            adata.obs["total_counts"] = matrix_counts
    else:
        matrix_counts = None
        n_genes = adata.obs["n_genes_by_counts"].to_numpy() if "n_genes_by_counts" in adata.obs else np.asarray((adata.X > 0).sum(axis=1)).ravel()
    transcript_counts = _ensure_transcript_counts(adata, matrix_counts)
    low_counts = transcript_counts < min_counts
    few_genes = n_genes < min_genes
    keep = ~low_counts & ~few_genes
    reason = np.full(n0, "pass", dtype=object)
    reason[low_counts] = "low_counts"
    reason[few_genes & ~low_counts] = "few_genes"
    reasons = {"low_counts": int(low_counts.sum()), "few_genes": int(few_genes.sum())}
    ctrl = _qc_control_counts(adata)
    if ctrl is not None and max_control_frac is not None:
        denom = np.clip(transcript_counts + ctrl, 1, None)
        frac = ctrl / denom
        adata.obs["control_frac"] = frac
        bad_ctrl = frac > max_control_frac
        keep &= ~bad_ctrl
        reasons["control_probes"] = int(bad_ctrl.sum())
        reason[bad_ctrl & (reason == "pass")] = "high_control_frac"
    if min_cell_area is not None and "cell_area" in adata.obs:
        bad = adata.obs["cell_area"].to_numpy() < min_cell_area
        keep &= ~bad
        reason[bad & (reason == "pass")] = "cell_area"
        reasons["cell_area"] = int(bad.sum())
    if max_cell_area is not None and "cell_area" in adata.obs:
        bad = adata.obs["cell_area"].to_numpy() > max_cell_area
        keep &= ~bad
        reason[bad & (reason == "pass")] = "cell_area"
        reasons["cell_area"] = reasons.get("cell_area", 0) + int(bad.sum())
    adata.obs[QC_PASS] = keep
    adata.obs[QC_FLAG] = pd.Categorical(np.where(keep, HIGH_QUALITY, LOW_QUALITY), categories=[HIGH_QUALITY, LOW_QUALITY])
    adata.obs[QC_REASON] = pd.Categorical(reason)
    adata.uns["qc"] = {"min_counts": min_counts, "min_genes": min_genes, "max_control_frac": max_control_frac,
                       "n_cells": int(n0), "n_low_quality": int((~keep).sum()), "reasons": reasons,
                       "count_column": "transcript_counts"}
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


def preprocess(adata: ad.AnnData, min_counts: int = 20, min_genes: int = 0, max_control_frac: float = 0.3, target_sum: float | None = None) -> ad.AnnData:
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
