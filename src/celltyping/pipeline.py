"""Config-driven end-to-end runs (knowledge build -> load -> preprocess -> annotate -> save).

Thin layer over the scanpy-style API (:mod:`celltyping.pp`, :mod:`celltyping.tl`) used by the CLI and the benchmark.
"""

from __future__ import annotations

import time
from pathlib import Path

import anndata as ad
import yaml

from . import pp, tl
from .io import read_dataset
from .knowledge.build import BuildParams, build_knowledge
from .knowledge.tree import CellTypeTree
from .methods import summarize_levels
from .utils import log


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_config_dir"] = str(Path(path).resolve().parent)
    return cfg


def _resolve(cfg: dict, p: str | None) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else Path(cfg["_config_dir"]) / p


def knowledge_path(cfg: dict) -> Path:
    out = _resolve(cfg, cfg.get("output_dir", "results")) / "knowledge"
    return out


def ensure_knowledge(cfg: dict, force: bool = False) -> CellTypeTree:
    """Build (or load cached) knowledge tree for the config's tissue and panel."""
    kcfg = cfg.get("knowledge", {})
    out = knowledge_path(cfg)
    from .knowledge.ontology import resolve_tissue

    tissue = resolve_tissue(cfg["tissue"])
    tree_file = out / f"{tissue.slug}_tree.json"
    if tree_file.exists() and not force:
        log.info("loading knowledge tree %s", tree_file)
        return CellTypeTree.load(tree_file)
    params = BuildParams(
        species=cfg.get("species", "human"),
        sources=tuple(kcfg.get("sources", ("CellMarker2", "PanglaoDB", "ASCT+B", "CellGuide"))),
        ancestor_levels=kcfg.get("ancestor_levels", 1),
        census=kcfg.get("census", True),
        census_min_cells=kcfg.get("census_min_cells", 200),
        census_min_datasets=kcfg.get("census_min_datasets", 2),
        census_min_frac=kcfg.get("census_min_frac", 0.001),
        census_disease=kcfg.get("census_disease", "normal"),
        **({"census_version": kcfg["census_version"]} if kcfg.get("census_version") else {}),
        corroborate_ancestor_hits=kcfg.get("corroborate_ancestor_hits", True),
        min_db_sources=kcfg.get("min_db_sources", 2),
        min_markers=kcfg.get("min_markers", 3),
        max_depth=kcfg.get("max_depth", 6),
        collapse_single_child=kcfg.get("collapse_single_child", True),
        propagate_shared_to_parent=kcfg.get("propagate_shared_to_parent", True),
    )
    tree, _, _ = build_knowledge(cfg["tissue"], panel=_resolve(cfg, cfg.get("panel")), params=params,
                                 overrides=_resolve(cfg, kcfg.get("overrides")), out_dir=out, force=force)
    return tree


def load_and_preprocess(cfg: dict, ds: dict, subsample: int | None = None, seed: int = 0) -> ad.AnnData:
    adata = read_dataset(ds["path"], ds.get("platform"), ds.get("name"))
    if subsample and subsample < adata.n_obs:
        import numpy as np

        rng = np.random.default_rng(seed)
        adata = adata[np.sort(rng.choice(adata.n_obs, subsample, replace=False))].copy()
        log.info("subsampled to %d cells", adata.n_obs)
    pcfg = cfg.get("preprocess", {})
    return pp.preprocess(adata, min_counts=pcfg.get("min_counts", 20), min_genes=pcfg.get("min_genes", 0),
                         max_control_frac=pcfg.get("max_control_frac", 0.3), target_sum=pcfg.get("target_sum"),
                         knn_smooth_k=pcfg.get("knn_smooth", 15), n_pcs=pcfg.get("n_pcs", 30))


def scoring_layer(cfg: dict) -> str | None:
    return "knn_smooth" if cfg.get("preprocess", {}).get("knn_smooth", 15) else None


def layer_key_present(adata: ad.AnnData) -> bool:
    return "knn_smooth" in adata.layers


def annotate_dataset(adata: ad.AnnData, tree: CellTypeTree, cfg: dict, methods: tuple[str, ...] = ("hier", "flat", "cluster")) -> dict[str, float]:
    """Run the requested annotators in place; returns wall-clock seconds per method."""
    acfg = cfg.get("annotate", {})
    method = acfg.get("method", "robust_z")
    top_k = acfg.get("top_k", 30)
    layer = scoring_layer(cfg) if layer_key_present(adata) else None
    sk = dict(acfg.get("scoring", {}) or {})  # passed to score_sets: scale, top_frac, n_null, ...
    timings = {}
    if "hier" in methods:
        t0 = time.perf_counter()
        h = acfg.get("hierarchical", {})
        tl.hierarchical(adata, tree, method=method, min_score=h.get("min_score", 0.5), min_margin=h.get("min_margin", 0.25),
                        min_cells=h.get("min_cells", 20), rescore_per_node=h.get("rescore_per_node", True), top_k=top_k,
                        subtree_signatures=h.get("subtree_signatures", True), parent_as_competitor=h.get("parent_as_competitor", True),
                        layer=layer, smooth=h.get("smooth"), subtree_mode=h.get("subtree_mode", "union"), **sk)
        timings["hier"] = time.perf_counter() - t0
    if "flat" in methods:
        t0 = time.perf_counter()
        f = acfg.get("flat", {})
        tl.flat(adata, tree, level=f.get("level", "leaves"), method=method, min_score=f.get("min_score", 0.5),
                min_margin=f.get("min_margin", 0.25), top_k=top_k, layer=layer, **sk)
        timings["flat"] = time.perf_counter() - t0
    if "rule" in methods or "rule_based" in methods:
        t0 = time.perf_counter()
        r = acfg.get("rule_based", acfg.get("rule", {}))
        tl.rule_based(adata, tree, level=r.get("level", "leaves"), n_markers=r.get("n_markers", 4),
                      markers_dict=r.get("markers_dict"), top_k=r.get("top_k"), key="rule")
        timings["rule"] = time.perf_counter() - t0
    if "cluster" in methods:
        t0 = time.perf_counter()
        c = acfg.get("cluster", {})
        tl.clusters(adata, tree, resolutions=tuple(c.get("resolutions", (0.5, 1.0))), level=c.get("level", "leaves"),
                    min_score=c.get("min_score", 0.2), min_margin=c.get("min_margin", 0.1), n_top_de=c.get("n_top_de", 25), top_k=top_k)
        timings["cluster"] = time.perf_counter() - t0
    adata.uns["timings_sec"] = timings
    return timings


def save_outputs(adata: ad.AnnData, cfg: dict, ds_name: str) -> Path:
    out = _resolve(cfg, cfg.get("output_dir", "results")) / ds_name
    out.mkdir(parents=True, exist_ok=True)
    # tables first (robust even if h5ad write fails on odd dtypes)
    label_cols = [c for c in adata.obs.columns if c.endswith("_label") or c.startswith("hier_level") or c.endswith("_id") or c.endswith("_depth")]
    ref_col = cfg.get("benchmark", {}).get("reference_column")
    extra = [c for c in ("x_centroid", "y_centroid", "transcript_counts", "total_counts", "qc_flag", "qc_reason", ref_col)
             if c and c in adata.obs and c not in label_cols]
    tab = adata.obs[label_cols + extra].copy()
    if "x_centroid" not in tab and "spatial" in adata.obsm:
        tab["x_centroid"], tab["y_centroid"] = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    tab.to_csv(out / "labels.csv")
    if any(c.startswith("hier_level") for c in adata.obs.columns):
        summarize_levels(adata).to_csv(out / "hier_level_counts.csv", index=False)
    for k, v in list(adata.uns.items()):
        if k.endswith("_labels") or k.endswith("_de"):
            v.to_csv(out / f"{k}.csv", index=False)
    for k in list(adata.uns):
        if k.startswith("rgg_"):
            del adata.uns[k]  # not h5ad-friendly across versions
    adata.write_h5ad(out / "annotated.h5ad")
    log.info("wrote outputs to %s", out)
    return out


def run(cfg: dict, datasets: list[str] | None = None, methods: tuple[str, ...] = ("hier", "flat", "cluster"),
        subsample: int | None = None, force_knowledge: bool = False) -> dict[str, Path]:
    tree = ensure_knowledge(cfg, force=force_knowledge)
    outputs = {}
    for ds in cfg["datasets"]:
        if datasets and ds["name"] not in datasets:
            continue
        log.info("=== dataset %s ===", ds["name"])
        adata = load_and_preprocess(cfg, ds, subsample=subsample)
        annotate_dataset(adata, tree, cfg, methods)
        outputs[ds["name"]] = save_outputs(adata, cfg, ds["name"])
    return outputs
