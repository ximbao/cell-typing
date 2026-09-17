"""``celltyping.bm``: benchmarking annotations against a reference (expert labels or another method).

    ref = ct.bm.reference(adata, "cell_type", {"Fibroblast": "CL:0000057", ...}, tree)
    summary, per_class, confusion = ct.bm.agreement(adata.obs["hier_id"], ref, tree)
    ct.bm.compare(adata, ref, tree, methods=["hier", "flat"])       # one summary row per method
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from .benchmark.harmonize import reference_ids, relation_table  # noqa: F401
from .benchmark.metrics import agreement, hierarchical_prf, label_entropy, marker_specificity, spatial_coherence  # noqa: F401
from .benchmark.run import prediction_columns
from .knowledge.tree import CellTypeTree
from .methods.scoring import UNASSIGNED


def reference(adata: ad.AnnData, column: str, mapping: dict[str, str], tree: CellTypeTree, ignore: list[str] | None = None,
              key: str | None = "ref_id") -> pd.Series:
    """Map free-text reference labels in ``obs[column]`` to tree node ids via ``mapping`` (label -> CL id; ids
    absent from the tree are lifted to their nearest tree ancestor). Stored in ``obs[key]`` when ``key`` is given."""
    ref = reference_ids(adata.obs[column], mapping, tree, ignore)
    if key:
        adata.obs[key] = ref.astype(object)
    return ref


def compare(adata: ad.AnnData, ref: pd.Series, tree: CellTypeTree, methods: tuple[str, ...] = ("hier", "flat", "cluster"),
            spatial_k: int = 10, marker_top_k: int = 30, layer: str | None = "auto") -> pd.DataFrame:
    """Summary table (one row per method found in ``obs``): lineage accuracy, coarser/unassigned/wrong fractions,
    hierarchical P/R/F1, macro-F1, ARI/NMI, spatial coherence and marker specificity. Low-quality cells
    (``obs['qc_pass'] == False``) are excluded."""
    layer = ("knn_smooth" if "knn_smooth" in adata.layers else None) if layer == "auto" else layer
    if "qc_pass" in adata.obs and not adata.obs["qc_pass"].all():  # evaluate high-quality cells only
        m = adata.obs["qc_pass"].astype(bool).to_numpy()
        adata, ref = adata[m].copy(), ref[m]
    coords = np.asarray(adata.obsm["spatial"], dtype=float) if "spatial" in adata.obsm else None
    rows = {}
    for name, col in prediction_columns(adata, tuple(methods)).items():
        pred = adata.obs[col].astype(object)
        summ, _, _ = agreement(pred, ref, tree)
        summ.update(label_entropy(pred, tree))
        if coords is not None:
            summ.update(spatial_coherence(coords, pred, tree, spatial_k))
        summ.update(marker_specificity(adata, pred, tree, top_k=marker_top_k, layer=layer))
        rows[name] = summ
    out = pd.DataFrame(rows).T
    out.index.name = "method"
    return out


def confusion(adata: ad.AnnData, ref: pd.Series, tree: CellTypeTree, key: str = "hier") -> pd.DataFrame:
    """Reference x prediction confusion matrix (labels) for one method (high-quality cells only)."""
    pred = adata.obs[f"{key}_id"].astype(object)
    if "qc_pass" in adata.obs:
        ref = ref.where(adata.obs["qc_pass"].astype(bool).to_numpy(), None)
    _, _, conf = agreement(pred, ref, tree)
    return conf


def labels_of(ids: pd.Series, tree: CellTypeTree) -> pd.Series:
    """Node ids -> labels (root / missing -> 'unassigned')."""
    return ids.map(lambda i: tree.nodes[i].label if i in tree.nodes and i != tree.root else UNASSIGNED)


__all__ = ["reference", "compare", "confusion", "agreement", "hierarchical_prf", "spatial_coherence", "marker_specificity",
           "label_entropy", "relation_table", "labels_of"]
