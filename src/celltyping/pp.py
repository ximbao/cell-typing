"""``celltyping.pp``: preprocessing (QC, normalisation, smoothing), scanpy-style.

The functions modify ``adata`` in place and also return it. ``qc`` does not remove cells: it adds
``obs['qc_pass']`` / ``obs['qc_flag']`` and every downstream step (smoothing, annotators, benchmark)
restricts itself to high-quality cells; pass ``filter=True`` to drop them instead.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

from .preprocess import embed  # noqa: F401
from .preprocess import harmonize_var_names as harmonize_genes  # noqa: F401
from .preprocess import HIGH_QUALITY, LOW_QUALITY, QC_FLAG, QC_PASS, QC_REASON, knn_smooth, normalize, qc_mask  # noqa: F401
from .preprocess import qc_filter as qc  # noqa: F401
from .utils import log


def is_raw_counts(adata: ad.AnnData, layer: str | None = None, n_check: int = 2000) -> bool:
    """Heuristic: does ``X`` (or ``layers[layer]``) hold raw integer counts?

    Checks a random subset of non-zero entries for integrality and looks at ``uns['normalized']``
    (set by :func:`normalize`) and ``uns['log1p']`` (set by scanpy).
    """
    if layer is None and ("normalized" in adata.uns or "log1p" in adata.uns):
        return False
    X = adata.layers[layer] if layer else adata.X
    if sp.issparse(X):
        data = X.data
    else:
        data = np.asarray(X).ravel()
        data = data[data != 0]
    if data.size == 0:
        return True
    rng = np.random.default_rng(0)
    sample = data[rng.choice(data.size, min(n_check, data.size), replace=False)]
    return bool(np.all(np.mod(sample, 1) == 0) and sample.max() >= 1)


def preprocess(adata: ad.AnnData, min_counts: int = 20, min_genes: int = 0, max_control_frac: float = 0.3,
               target_sum: float | None = None, knn_smooth_k: int | None = 15, n_pcs: int = 30,
               random_state: int = 0, filter: bool = False) -> ad.AnnData:
    """Full default preprocessing for imaging-based ST: harmonise gene symbols, QC flag (``obs['qc_flag']``;
    ``transcript_counts < min_counts`` -> ``'low_quality'``), ``normalize_total`` + ``log1p`` (raw counts kept in
    ``layers['counts']``), then kNN expression smoothing over high-quality cells into ``layers['knn_smooth']``
    (skip with ``knn_smooth_k=None``). Low-quality cells are kept unless ``filter=True``."""
    adata = harmonize_genes(adata)
    if is_raw_counts(adata):
        adata = qc(adata, min_counts=min_counts, min_genes=min_genes, max_control_frac=max_control_frac, filter=filter)
        normalize(adata, target_sum=target_sum)
    else:
        log.info("X does not look like raw counts; skipping normalisation")
        if QC_PASS not in adata.obs:
            adata = qc(adata, min_counts=min_counts, min_genes=min_genes, max_control_frac=max_control_frac, filter=filter)
    if knn_smooth_k:
        knn_smooth(adata, k=int(knn_smooth_k), n_pcs=n_pcs, random_state=random_state)
    return adata


__all__ = ["qc", "qc_mask", "normalize", "harmonize_genes", "knn_smooth", "embed", "is_raw_counts", "preprocess",
           "QC_PASS", "QC_FLAG", "QC_REASON", "HIGH_QUALITY", "LOW_QUALITY"]
