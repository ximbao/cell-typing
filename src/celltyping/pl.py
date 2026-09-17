"""``celltyping.pl``: plotting.

Every function returns the matplotlib ``Axes`` (or ``Figure`` for multi-panel plots) and accepts
``ax=`` and ``save=`` like scanpy; nothing is shown automatically (call ``plt.show()``).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmark.report import draw_confusion
from .knowledge.tree import CellTypeTree
from .methods.scoring import UNASSIGNED
from .preprocess import LOW_QUALITY

_PALETTE = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors


def _finish(fig, save):
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")


def palette(labels, tree: CellTypeTree | None = None) -> dict[str, tuple]:
    """Stable label -> colour mapping (unassigned is grey). With a ``tree`` the order follows the tree (BFS),
    so sibling types get neighbouring colours consistently across plots."""
    labels = [str(l) for l in labels]
    if tree is not None:
        order = [n.label for n in tree.iter_bfs() if n.label in set(labels)]
        order += [l for l in pd.unique(pd.Series(labels)) if l not in set(order)]
    else:
        order = list(pd.Series(labels).value_counts().index)
    out = {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(l for l in order if l not in (UNASSIGNED, LOW_QUALITY))}
    out[UNASSIGNED] = (0.85, 0.85, 0.85)
    out[LOW_QUALITY] = (0.94, 0.94, 0.94)
    return out


# --------------------------------------------------------------------------- tree
def _k(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 10_000 else (f"{n / 1000:.1f}k" if n >= 1000 else str(n))


def tree(tree: CellTypeTree, markers: int = 4, census: bool = True, ax=None, save=None, fontsize: int = 8):
    """Draw the cell-type hierarchy left-to-right; node text shows the label, the number of panel markers
    (and Census cell counts when available) and the top markers."""
    leaves: list[str] = []
    stack = [tree.root]
    while stack:  # DFS so that leaves of one subtree are contiguous
        cur = stack.pop()
        kids = tree.nodes[cur].children
        if kids:
            stack.extend(reversed(list(kids)))
        else:
            leaves.append(cur)
    y_of: dict[str, float] = {l: float(i) for i, l in enumerate(leaves)}

    def _y(nid: str) -> float:
        if nid in y_of:
            return y_of[nid]
        ys = [_y(c) for c in tree.nodes[nid].children]
        y_of[nid] = (min(ys) + max(ys)) / 2
        return y_of[nid]

    _y(tree.root)
    depth = {nid: tree.depth(nid) for nid in tree.nodes}
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6 * (tree.max_depth() + 1) + 1, 0.34 * len(leaves) + 1))
    else:
        fig = ax.figure
    for n in tree.iter_bfs():
        x, y = depth[n.id], y_of[n.id]
        if n.parent:
            px, py = depth[n.parent], y_of[n.parent]
            ax.plot([px, px, x], [py, y, y], color="0.6", lw=0.8, zorder=1)
        pm = sorted(n.panel_markers.values(), key=lambda m: -m.weight)
        txt = n.label if n.id == tree.root else f"{n.label} ({len(pm)})"
        if census and n.census:
            txt += f" \u00b7 {_k(n.census['n_cells'])} Census cells"
        if markers and pm:
            txt += "\n" + ", ".join(m.gene for m in pm[:markers])
        ax.scatter([x], [y], s=28, color="#4c72b0" if n.seed else "0.5", zorder=2)
        ax.text(x + 0.05, y, txt, va="center", fontsize=fontsize, zorder=3)
    ax.set_xlim(-0.2, tree.max_depth() + 1.2)
    ax.set_ylim(len(leaves) - 0.5, -0.5)
    ax.axis("off")
    ax.set_title(f"{tree.meta.get('tissue', {}).get('label', '')} cell types ({len(tree)} nodes, {len(leaves)} leaves)")
    _finish(fig, save)
    return ax


# --------------------------------------------------------------------------- spatial
def spatial(adata: ad.AnnData, key: str = "hier_label", ax=None, size: float = 0.4, max_points: int = 200_000, legend: bool = True,
            colors: dict | None = None, tree: CellTypeTree | None = None, title: str | None = None, save=None, random_state: int = 0):
    """Scatter of cells in ``obsm['spatial']`` coloured by ``obs[key]`` (categorical) or a continuous ``obs`` column."""
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(adata.n_obs, min(adata.n_obs, max_points), replace=False))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6.5))
    else:
        fig = ax.figure
    vals = adata.obs[key]
    if pd.api.types.is_numeric_dtype(vals):
        sc = ax.scatter(xy[idx, 0], xy[idx, 1], c=vals.to_numpy()[idx], s=size, linewidths=0, cmap="viridis", rasterized=True)
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label=key)
    else:
        labs = vals.astype(str).to_numpy()
        colors = colors or palette(labs, tree)
        c = np.array([colors.get(l, (0.5, 0.5, 0.5)) for l in labs[idx]])
        ax.scatter(xy[idx, 0], xy[idx, 1], c=c, s=size, linewidths=0, rasterized=True)
        if legend:
            counts = pd.Series(labs).value_counts()
            handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[l], label=f"{l} ({counts[l]:,})", markersize=6)
                       for l in counts.index[:30] if l in colors]
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or key)
    _finish(fig, save)
    return ax


# --------------------------------------------------------------------------- composition
def composition(adata: ad.AnnData, key: str = "hier", groupby: str | None = None, ax=None, tree: CellTypeTree | None = None, save=None):
    """Per-level cell-type composition of a hierarchical run (one stacked bar per level), or -- with
    ``groupby`` -- the composition of ``obs[key+'_label']`` per group (e.g. sample or region)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    if groupby is None:
        cols = sorted([c for c in adata.obs.columns if c.startswith(f"{key}_level")], key=lambda c: int(c.replace(f"{key}_level", "")))
        if not cols:
            cols = [f"{key}_label"]
        tab = pd.DataFrame({c.replace(f"{key}_level", "level "): adata.obs[c].astype(str).value_counts(normalize=True) for c in cols}).fillna(0).T
    else:
        tab = pd.crosstab(adata.obs[groupby], adata.obs[f"{key}_label"].astype(str), normalize="index")
    colors = palette(tab.columns, tree)
    tab = tab[sorted(tab.columns, key=lambda c: -tab[c].sum())]
    tab.plot.barh(stacked=True, ax=ax, color=[colors[c] for c in tab.columns], width=0.8, legend=False)
    ax.set_xlabel("fraction of cells")
    ax.set_xlim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False, ncol=1 + len(tab.columns) // 25)
    _finish(fig, save)
    return ax


# --------------------------------------------------------------------------- scores
def scores(adata: ad.AnnData, key: str = "hier", groupby: str | None = None, nodes: list[str] | None = None, tree: CellTypeTree | None = None,
           ax=None, save=None, cmap: str = "RdBu_r", vmax: float | None = None):
    """Heat-map of mean signature scores (``obsm[key+'_scores']``) per predicted label (rows) x scored node (columns).
    Highlights whether labels are supported by their own signature and where confusions arise."""
    S = adata.obsm[f"{key}_scores"] if f"{key}_scores" in adata.obsm else adata.obsm[key]
    S = pd.DataFrame(S, index=adata.obs_names) if not isinstance(S, pd.DataFrame) else S
    if tree is not None:
        S = S.rename(columns={c: tree.nodes[c].label for c in S.columns if c in tree.nodes})
    if nodes:
        S = S[[c for c in S.columns if c in set(nodes)]]
    groupby = groupby or f"{key}_label"
    M = S.groupby(adata.obs[groupby].astype(str).to_numpy()).mean()
    M = M.loc[~M.index.isin([UNASSIGNED, LOW_QUALITY])]
    M = M.dropna(axis=1, how="all")
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.35 * M.shape[1] + 3, 0.3 * M.shape[0] + 2))
    else:
        fig = ax.figure
    v = vmax or float(np.nanpercentile(np.abs(M.to_numpy()), 98))
    im = ax.imshow(M.to_numpy(), cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index, fontsize=7)
    ax.set_xlabel("signature")
    ax.set_ylabel(groupby)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="mean score")
    _finish(fig, save)
    return ax


# --------------------------------------------------------------------------- confusion
def confusion(conf: pd.DataFrame, ax=None, normalize: str = "row", title: str = "", save=None):
    """Reference x prediction confusion matrix (from :func:`celltyping.bm.confusion` / :func:`celltyping.bm.agreement`)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(conf.columns) + 3), max(4, 0.4 * len(conf.index) + 2)))
    else:
        fig = ax.figure
    draw_confusion(conf, ax, title, normalize)
    _finish(fig, save)
    return ax


def markers(adata: ad.AnnData, tree: CellTypeTree, groupby: str = "hier_label", n_markers: int = 4, layer: str | None = "auto",
            save=None, **kwargs):
    """Dot-plot of the top tree markers of each predicted type, grouped by ``obs[groupby]`` (uses ``scanpy.pl.dotplot``)."""
    import scanpy as sc

    layer = ("knn_smooth" if "knn_smooth" in adata.layers else None) if layer == "auto" else layer
    present = set(adata.obs[groupby].astype(str))
    var_groups = {}
    for n in tree.iter_bfs():
        if n.label in present:
            pm = sorted(n.panel_markers.values(), key=lambda m: -m.weight)
            genes = [m.gene for m in pm[:n_markers] if m.gene in adata.var_names]
            if genes:
                var_groups[n.label] = genes
    dp = sc.pl.dotplot(adata, var_groups, groupby=groupby, layer=layer, show=False, return_fig=True, **kwargs)
    if save:
        dp.savefig(save, dpi=150, bbox_inches="tight")
    return dp


__all__ = ["tree", "spatial", "composition", "scores", "confusion", "markers", "palette"]
