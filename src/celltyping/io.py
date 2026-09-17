"""Readers for Xenium and CosMx outputs -> AnnData with ``obsm['spatial']`` and raw counts in ``X``."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import log

CONTROL_PREFIXES = ("negcontrol", "unassigned", "blank", "deprecated", "negprb", "systemcontrol", "falsecode")


def _split_controls(adata: ad.AnnData) -> ad.AnnData:
    """Move control probes/codewords out of ``var`` into per-cell ``obs`` totals."""
    names = pd.Index(adata.var_names.astype(str))
    ft = adata.var["feature_types"].astype(str).str.lower() if "feature_types" in adata.var else pd.Series("gene expression", index=adata.var_names)
    is_ctrl = names.str.lower().str.startswith(CONTROL_PREFIXES) | ~ft.str.contains("gene").values
    if is_ctrl.any():
        ctrl = adata[:, is_ctrl].X
        adata.obs["control_counts"] = np.asarray(ctrl.sum(axis=1)).ravel()
        adata = adata[:, ~is_ctrl].copy()
    else:
        adata.obs["control_counts"] = 0.0
    return adata


def read_xenium(path: str | Path, region_name: str | None = None) -> ad.AnnData:
    """Read a Xenium output bundle (folder with ``cell_feature_matrix.h5`` and ``cells.parquet``)."""
    import scanpy as sc

    path = Path(path)
    h5 = path / "cell_feature_matrix.h5"
    adata = sc.read_10x_h5(h5, gex_only=False)
    adata.var_names_make_unique()
    adata = _split_controls(adata)
    cells_pq = path / "cells.parquet"
    cells = pd.read_parquet(cells_pq) if cells_pq.exists() else pd.read_csv(path / "cells.csv.gz")
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.set_index("cell_id").reindex(adata.obs_names)
    for c in ("x_centroid", "y_centroid", "transcript_counts", "cell_area", "nucleus_area", "nucleus_count"):
        if c in cells:
            adata.obs[c] = cells[c].values
    adata.obsm["spatial"] = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    adata.obs["region"] = region_name or path.name
    adata.uns["platform"] = "xenium"
    adata.uns["source_path"] = str(path)
    log.info("Xenium %s: %d cells x %d genes", path.name, *adata.shape)
    return adata


def read_cosmx(path: str | Path, region_name: str | None = None) -> ad.AnnData:
    """Read CosMx flat files (``*exprMat_file.csv[.gz]`` + ``*metadata_file.csv[.gz]``) from a folder."""
    path = Path(path)
    expr_f = next(iter(sorted(path.glob("*exprMat_file.csv*"))), None)
    meta_f = next(iter(sorted(path.glob("*metadata_file.csv*"))), None)
    if expr_f is None or meta_f is None:
        raise FileNotFoundError(f"CosMx flat files not found under {path}")
    expr = pd.read_csv(expr_f)
    meta = pd.read_csv(meta_f)
    key = ["fov", "cell_ID"]
    expr = expr[expr["cell_ID"] != 0]  # drop extracellular counts
    expr["cell"] = expr["fov"].astype(str) + "_" + expr["cell_ID"].astype(str)
    meta["cell"] = meta["fov"].astype(str) + "_" + meta["cell_ID"].astype(str)
    expr = expr.set_index("cell").drop(columns=key)
    genes = [c for c in expr.columns if not c.lower().startswith(CONTROL_PREFIXES)]
    ctrls = [c for c in expr.columns if c not in genes]
    X = sp.csr_matrix(expr[genes].to_numpy(dtype=np.float32))
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=expr.index), var=pd.DataFrame(index=genes))
    adata.obs["control_counts"] = expr[ctrls].sum(axis=1).values if ctrls else 0.0
    meta = meta.set_index("cell").reindex(adata.obs_names)
    for c in meta.columns:
        if c not in key:
            adata.obs[c] = meta[c].values
    xcol = "CenterX_global_px" if "CenterX_global_px" in meta else "CenterX_local_px"
    ycol = xcol.replace("X", "Y")
    adata.obsm["spatial"] = meta[[xcol, ycol]].to_numpy(dtype=float)
    adata.obs["region"] = region_name or path.name
    adata.uns["platform"] = "cosmx"
    adata.uns["source_path"] = str(path)
    log.info("CosMx %s: %d cells x %d genes", path.name, *adata.shape)
    return adata


def read_dataset(path: str | Path, platform: str | None = None, region_name: str | None = None) -> ad.AnnData:
    """Dispatch on ``platform`` (``xenium``/``cosmx``/``h5ad``) or infer it from the folder contents."""
    path = Path(path)
    if platform is None:
        if path.suffix == ".h5ad":
            platform = "h5ad"
        elif (path / "cell_feature_matrix.h5").exists():
            platform = "xenium"
        elif any(path.glob("*exprMat_file.csv*")):
            platform = "cosmx"
        else:
            raise ValueError(f"cannot infer platform for {path}")
    if platform == "h5ad":
        adata = ad.read_h5ad(path)
        if region_name:
            adata.obs["region"] = region_name
        return adata
    if platform == "xenium":
        return read_xenium(path, region_name)
    if platform == "cosmx":
        return read_cosmx(path, region_name)
    raise ValueError(f"unknown platform {platform}")
