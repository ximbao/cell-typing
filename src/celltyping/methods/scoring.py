"""Per-cell marker-set scoring shared by the flat and hierarchical annotators."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from ..preprocess import gene_sd_floor, zscore_matrix
from ..utils import log

UNASSIGNED = "unassigned"


def _restrict_sets(sets: dict[str, dict[str, float]], genes: set[str]) -> dict[str, dict[str, float]]:
    out = {}
    for k, gw in sets.items():
        sub = {g: w for g, w in gw.items() if g in genes}
        if sub:
            out[k] = sub
    return out


def _top_frac_mean(Z: np.ndarray, top_frac: float, min_top: int) -> np.ndarray:
    """Per row, mean of the largest ``max(min_top, ceil(top_frac * n))`` values."""
    n = Z.shape[1]
    k = int(min(n, max(min_top, np.ceil(top_frac * n))))
    if k >= n:
        return Z.mean(axis=1)
    part = np.partition(Z, n - k, axis=1)[:, n - k:]
    return part.mean(axis=1)


def _detection_fraction(adata: ad.AnnData) -> np.ndarray:
    """Fraction of cells with a non-zero count per gene (from ``layers['counts']`` if present, else ``X``); cached in ``var``."""
    if "detection_frac" in adata.var and adata.var["detection_frac"].notna().all():
        return adata.var["detection_frac"].to_numpy(dtype=float)
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if hasattr(X, "getnnz"):
        det = np.asarray(X.getnnz(axis=0), dtype=float) / X.shape[0] if X.format == "csc" else np.asarray((X != 0).sum(axis=0)).ravel() / X.shape[0]
    else:
        det = (np.asarray(X) != 0).mean(axis=0)
    if not adata.is_view:
        adata.var["detection_frac"] = det
    return det


def score_sets(adata: ad.AnnData, sets: dict[str, dict[str, float]], method: str = "robust_z", clip: float = 10.0,
               top_frac: float = 0.34, min_top: int = 3, n_null: int = 30, layer: str | None = None,
               random_state: int = 0, scale: str = "center", sd_floor: bool = True, match_detection: bool = False) -> pd.DataFrame:
    """Score every cell in ``adata`` against each weighted gene set.

    methods
    -------
    ``robust_z`` (default) weighted z-scores; per cell, the mean of the top ``top_frac`` of the set's
                 genes (at least ``min_top``) so that a cell expressing only part of a heterogeneous
                 signature still scores high. Calibrated against ``n_null`` random gene sets of the
                 same size: the result is in null-standard-deviation units, comparable across set sizes.
    ``mean_z``   weighted mean of per-gene z-scores (computed over the cells passed in).
    ``ulm``      decoupler univariate linear model (t-values).
    ``aucell``   decoupler AUCell (rank-based, weights ignored).
    ``layer``    score from ``adata.layers[layer]`` (e.g. kNN-smoothed expression) instead of ``X``.
    ``scale``    ``'center'`` (default): per-gene mean-centred log expression, no per-gene rescaling;
                 ``'z'``: per-gene z-scores. In sparse imaging data z-scores let rarely detected genes
                 dominate a set from a single count, centring is markedly more robust.
    ``sd_floor`` (``scale='z'`` only) regularise per-gene SDs with the median gene SD (see ``preprocess.zscore_matrix``).
    ``match_detection`` draw null genes with the same detection-rate profile as the set. Off by default:
                 when one cell type dominates the sample, highly detected genes are that type's genes and
                 the matched null becomes biased against it.
    Returns a DataFrame (cells x sets); sets with no gene in ``adata`` are dropped.
    """
    sets = _restrict_sets(sets, set(adata.var_names))
    if not sets:
        return pd.DataFrame(index=adata.obs_names)
    if method == "robust_z":
        rng = np.random.default_rng(random_state)
        s0 = gene_sd_floor(adata, layer=layer) if (sd_floor and scale == "z") else None
        genes = sorted({g for gw in sets.values() for g in gw})
        Z = zscore_matrix(adata, genes, clip=clip, layer=layer, s0=s0, scale=scale)
        gi = {g: i for i, g in enumerate(genes)}
        # Null: random gene sets, optionally matched on detection rate (stratified background pool).
        det = _detection_fraction(adata)
        n_bg_total = sum(g not in gi for g in adata.var_names)
        n_bins = int(min(10, max(1, n_bg_total // 30))) if match_detection else 1  # small panels: too few genes to stratify
        edges = np.quantile(np.log10(det + 1e-4), np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1 else np.array([])
        gene_bin = dict(zip(adata.var_names, np.digitize(np.log10(det + 1e-4), edges)))
        bg_by_bin: dict[int, list[str]] = {}
        for g in adata.var_names:
            if g not in gi:
                bg_by_bin.setdefault(gene_bin[g], []).append(g)
        per_bin = max(80, 600 // n_bins)
        bg: list[str] = []
        bin_cols: dict[int, np.ndarray] = {}
        for b in range(n_bins):
            pool = bg_by_bin.get(b, [])
            if not pool:  # every gene of this bin is a marker: borrow the nearest populated bin
                near = min((bb for bb in bg_by_bin if bg_by_bin[bb]), key=lambda bb: abs(bb - b), default=None)
                pool = bg_by_bin.get(near, []) if near is not None else genes
            pick = list(rng.choice(pool, min(len(pool), per_bin), replace=False)) if len(pool) > per_bin else list(pool)
            bin_cols[b] = np.arange(len(bg), len(bg) + len(pick))
            bg.extend(pick)
        Zbg = zscore_matrix(adata, bg, clip=clip, layer=layer, s0=s0, scale=scale)
        null_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

        def _null(bins: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
            if bins not in null_cache:
                draws = np.empty((adata.n_obs, n_null), dtype=np.float32)
                for r in range(n_null):
                    cols = np.array([rng.choice(bin_cols[b]) for b in bins])
                    draws[:, r] = _top_frac_mean(Zbg[:, cols], top_frac, min_top)
                mu, sd = draws.mean(axis=1), draws.std(axis=1)
                floor = max(1e-3, 0.25 * float(np.median(sd)))  # avoid blow-ups for near-constant cells
                sd = np.maximum(sd, floor)
                null_cache[bins] = (mu, sd)
            return null_cache[bins]

        out = np.zeros((adata.n_obs, len(sets)), dtype=np.float32)
        for j, (k, gw) in enumerate(sets.items()):
            idx = np.array([gi[g] for g in gw])
            w = np.array([gw[g] for g in gw], dtype=np.float32)
            w = w / w.mean()  # weights scale around 1 so z units are preserved
            obs = _top_frac_mean(Z[:, idx] * w, top_frac, min_top)
            mu, sd = _null(tuple(sorted(gene_bin[g] for g in gw)))
            out[:, j] = (obs - mu) / sd
        return pd.DataFrame(out, index=adata.obs_names, columns=list(sets))
    if method == "mean_z":
        genes = sorted({g for gw in sets.values() for g in gw})
        Z = zscore_matrix(adata, genes, clip=clip, layer=layer, scale=scale)
        gi = {g: i for i, g in enumerate(genes)}
        out = np.zeros((adata.n_obs, len(sets)), dtype=np.float32)
        for j, (k, gw) in enumerate(sets.items()):
            idx = np.array([gi[g] for g in gw])
            w = np.array([gw[g] for g in gw], dtype=np.float32)
            w = w / w.sum()
            out[:, j] = Z[:, idx] @ w
        return pd.DataFrame(out, index=adata.obs_names, columns=list(sets))
    if method in ("ulm", "aucell"):
        return _decoupler_scores(adata, sets, method, layer=layer)
    raise ValueError(f"unknown scoring method {method}")


def _decoupler_scores(adata: ad.AnnData, sets: dict[str, dict[str, float]], method: str, layer: str | None = None) -> pd.DataFrame:
    import decoupler as dc

    net = pd.DataFrame([(k, g, w) for k, gw in sets.items() for g, w in gw.items()], columns=["source", "target", "weight"])
    a = ad.AnnData(X=(adata.layers[layer] if layer else adata.X), obs=adata.obs[[]].copy(), var=adata.var[[]].copy())
    if method == "ulm":
        if hasattr(dc, "run_ulm"):
            dc.run_ulm(a, net, min_n=1, use_raw=False, verbose=False)
            return a.obsm["ulm_estimate"]
        dc.mt.ulm(a, net, tmin=1, verbose=False)  # decoupler >= 2
        return a.obsm["score_ulm"]
    if hasattr(dc, "run_aucell"):
        dc.run_aucell(a, net, min_n=1, use_raw=False, verbose=False)
        return a.obsm["aucell_estimate"]
    dc.mt.aucell(a, net, tmin=1, verbose=False)
    return a.obsm["score_aucell"]


def assign(scores: pd.DataFrame, min_score: float = 0.0, min_margin: float = 0.0) -> pd.DataFrame:
    """Argmax assignment with confidence gating.

    Returns columns ``label`` (set id or ``unassigned``), ``score``, ``runner_up``, ``margin``.
    """
    if scores.shape[1] == 0:
        return pd.DataFrame({"label": UNASSIGNED, "score": np.nan, "runner_up": None, "margin": np.nan}, index=scores.index)
    vals = scores.to_numpy()
    order = np.argsort(-vals, axis=1)
    best = order[:, 0]
    best_score = vals[np.arange(len(vals)), best]
    if scores.shape[1] > 1:
        second = order[:, 1]
        second_score = vals[np.arange(len(vals)), second]
        runner = scores.columns.to_numpy()[second]
    else:
        second_score = np.full(len(vals), -np.inf)
        runner = np.array([None] * len(vals), dtype=object)
    margin = best_score - second_score
    ok = (best_score >= min_score) & (np.isinf(second_score) | (margin >= min_margin))
    label = np.where(ok, scores.columns.to_numpy()[best], UNASSIGNED)
    return pd.DataFrame({"label": label, "score": best_score, "runner_up": runner, "margin": np.where(np.isinf(margin), np.nan, margin)}, index=scores.index)


def smooth_labels(adata: ad.AnnData, label_key: str, conf_key: str | None = None, k: int = 10, min_frac: float = 0.6,
                  max_conf: float | None = None, out_key: str | None = None) -> pd.Series:
    """kNN majority-vote smoothing over ``obsm['spatial']``.

    A cell is relabelled when at least ``min_frac`` of its ``k`` nearest neighbours share a
    label different from its own and (if ``conf_key``/``max_conf`` given) its confidence is low.
    """
    from sklearn.neighbors import NearestNeighbors

    xy = adata.obsm["spatial"]
    labels = adata.obs[label_key].astype(str).to_numpy()
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, idx = nn.kneighbors(xy)
    idx = idx[:, 1:]
    new = labels.copy()
    n_changed = 0
    for i in range(len(labels)):
        neigh = labels[idx[i]]
        vals, counts = np.unique(neigh, return_counts=True)
        j = counts.argmax()
        if vals[j] != labels[i] and counts[j] / k >= min_frac:
            if conf_key is not None and max_conf is not None and adata.obs[conf_key].iloc[i] > max_conf:
                continue
            new[i] = vals[j]
            n_changed += 1
    log.info("smoothing: relabelled %d / %d cells", n_changed, len(labels))
    out_key = out_key or f"{label_key}_smooth"
    adata.obs[out_key] = pd.Categorical(new)
    return adata.obs[out_key]
