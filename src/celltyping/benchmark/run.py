"""Benchmark driver: annotate (or reuse a saved run), compare every method with the expert reference, write tables/figures."""

from __future__ import annotations

import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ..methods.scoring import UNASSIGNED
from ..knowledge.tree import CellTypeTree
from ..pipeline import _resolve, annotate_dataset, ensure_knowledge, load_and_preprocess, save_outputs, scoring_layer
from ..utils import log
from .harmonize import reference_ids
from .metrics import agreement, label_entropy, marker_specificity, spatial_coherence
from .report import plot_confusion, plot_spatial, plot_summary, write_markdown

METHOD_PATTERNS = {"hier": r"^hier_id$", "flat": r"^flat_id$", "rule": r"^rule_id$", "cluster": r"^cluster_r[\d.]+_id$"}


def prediction_columns(adata: ad.AnnData, methods: tuple[str, ...]) -> dict[str, str]:
    """Method display name -> obs column holding tree node ids."""
    cols = {}
    for m in methods:
        pat = re.compile(METHOD_PATTERNS[m])
        for c in adata.obs.columns:
            if pat.match(c):
                cols[c[: -len("_id")]] = c
    return cols


def _tree_txt(cfg: dict) -> str:
    from ..knowledge.ontology import resolve_tissue

    out = _resolve(cfg, cfg.get("output_dir", "results")) / "knowledge"
    p = out / f"{resolve_tissue(cfg['tissue']).slug}_tree.txt"
    return p.read_text() if p.exists() else ""


def run_benchmark(cfg: dict, dataset: str, methods: tuple[str, ...] = ("hier", "flat", "cluster"), subsample: int | None = None,
                  reuse: bool = True, force_knowledge: bool = False) -> Path:
    bcfg = cfg.get("benchmark")
    if not bcfg or "reference_column" not in bcfg:
        raise ValueError("config needs a 'benchmark' section with 'reference_column' and 'reference_map'")
    ds = next(d for d in cfg["datasets"] if d["name"] == dataset)
    tree = ensure_knowledge(cfg, force=force_knowledge)
    run_dir = _resolve(cfg, cfg.get("output_dir", "results")) / dataset
    h5 = run_dir / "annotated.h5ad"
    if reuse and h5.exists() and subsample is None:
        log.info("reusing annotated run %s", h5)
        adata = ad.read_h5ad(h5)
        missing = [m for m in methods if not prediction_columns(adata, (m,))]
        if missing:
            log.info("methods %s missing in saved run; annotating them", missing)
            annotate_dataset(adata, tree, cfg, tuple(missing))
            save_outputs(adata, cfg, dataset)
    else:
        adata = load_and_preprocess(cfg, ds, subsample=subsample)
        annotate_dataset(adata, tree, cfg, methods)
        save_outputs(adata, cfg, dataset)

    out = run_dir / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    if "qc_pass" in adata.obs and not adata.obs["qc_pass"].all():
        n_low = int((~adata.obs["qc_pass"].astype(bool)).sum())
        adata = adata[adata.obs["qc_pass"].astype(bool).to_numpy()].copy()
        log.info("benchmark: %d low-quality cells excluded; evaluating %d high-quality cells", n_low, adata.n_obs)
        notes_qc = [f"{n_low} low-quality cells (obs['qc_flag']) were excluded from the evaluation."]
    else:
        notes_qc = []
    ref_col = bcfg["reference_column"]
    if ref_col not in adata.obs:
        raise KeyError(f"reference column '{ref_col}' not in obs")
    ref = reference_ids(adata.obs[ref_col], bcfg.get("reference_map", {}), tree, bcfg.get("ignore_labels"))
    n_ref = int(ref.notna().sum())
    log.info("reference: %d / %d cells mapped onto %d tree nodes", n_ref, adata.n_obs, ref.dropna().nunique())

    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    k_sp = int(bcfg.get("spatial_k", 10))
    layer = scoring_layer(cfg) if "knn_smooth" in adata.layers else None
    top_k = cfg.get("annotate", {}).get("top_k", 30)
    timings = adata.uns.get("timings_sec", {})

    rows, per_class, notes = {}, {}, list(notes_qc)
    # the reference itself, for label-free metrics (spatial coherence / marker specificity anchors)
    ref_full = ref.where(ref.notna(), UNASSIGNED)
    rows["reference"] = {"n_eval": n_ref, **label_entropy(ref_full, tree), **spatial_coherence(coords, ref_full, tree, k_sp),
                         **marker_specificity(adata, ref_full, tree, top_k=top_k, layer=layer)}
    for name, col in prediction_columns(adata, methods).items():
        pred = adata.obs[col].astype(object)
        summ, pc, conf = agreement(pred, ref, tree)
        summ.update(label_entropy(pred, tree))
        summ.update(spatial_coherence(coords, pred, tree, k_sp))
        summ.update(marker_specificity(adata, pred, tree, top_k=top_k, layer=layer))
        base = name.split("_r")[0] if name.startswith("cluster") else name
        summ["runtime_sec"] = float(timings.get(base, np.nan))
        rows[name] = summ
        per_class[name] = pc
        pc.to_csv(out / f"per_class_{name}.csv", index=False)
        conf.to_csv(out / f"confusion_{name}.csv")
        plot_confusion(conf, out / f"confusion_{name}.png", f"{dataset}: {name} vs expert")
        log.info("%-14s lineage_acc=%.3f exact=%.3f coarser=%.3f unassigned=%.3f wrong=%.3f hF1=%.3f ARI=%.3f",
                 name, summ["lineage_acc"], summ["exact_acc"], summ["coarser"], summ["unassigned"], summ["wrong"], summ["hF1"], summ["ARI_assigned"])
    summary = pd.DataFrame(rows).T
    summary.index.name = "method"
    summary.to_csv(out / "summary.csv")
    plot_summary(summary, out / "summary.png")

    # spatial maps: reference (expert labels as given) + each method's label
    lab_of = lambda s: s.map(lambda i: tree.nodes[i].label if i in tree.nodes and i != tree.root else UNASSIGNED)  # noqa: E731
    maps = {"expert (mapped)": lab_of(ref_full)}
    for name, col in prediction_columns(adata, methods).items():
        maps[name] = lab_of(adata.obs[col].astype(object))
    plot_spatial(coords, maps, out / "spatial_labels.png")

    unmapped = sorted(set(adata.obs[ref_col].astype(str).unique()) - set(bcfg.get("reference_map", {})) - set(bcfg.get("ignore_labels") or []))
    if unmapped:
        notes.append(f"reference labels without a mapping (excluded from agreement): {unmapped}")
    notes.append(f"{n_ref} of {adata.n_obs} cells have a mapped reference label; reference_map: "
                 + json.dumps({k: tree.nodes[v].label if v in tree else v for k, v in bcfg.get('reference_map', {}).items()}))
    if timings:
        notes.append("runtime (s): " + ", ".join(f"{k}={v:.0f}" for k, v in timings.items()))
    write_markdown(out, dataset, summary, per_class, _tree_txt(cfg), notes)
    log.info("benchmark written to %s", out)
    return out
