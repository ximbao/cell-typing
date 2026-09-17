"""Figures and a markdown report for a benchmark run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..methods.scoring import UNASSIGNED  # noqa: E402


def draw_confusion(conf: pd.DataFrame, ax, title: str = "", normalize: str = "row", xlabel: str = "predicted (projected onto reference classes)",
                   ylabel: str = "expert reference") -> None:
    """Draw a reference x prediction confusion matrix on ``ax`` (reference classes first, then extra columns)."""
    mat = conf.to_numpy(dtype=float)
    if normalize == "row":
        mat = mat / np.clip(mat.sum(axis=1, keepdims=True), 1, None)
    cols = list(conf.columns)
    ordered = [c for c in conf.index if c in cols] + sorted([c for c in cols if c not in set(conf.index) and c != UNASSIGNED],
                                                            key=lambda c: -conf[c].sum())
    if UNASSIGNED in cols:
        ordered.append(UNASSIGNED)
    mat = mat[:, [cols.index(c) for c in ordered]]
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if normalize == "row" else None, aspect="auto")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(conf.index)))
    ax.set_yticklabels([f"{r} (n={int(conf.loc[r].sum())})" for r in conf.index], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] >= 0.05:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if mat[i, j] > 0.6 else "black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="fraction of reference class" if normalize == "row" else "cells")


def plot_confusion(conf: pd.DataFrame, path: Path, title: str, normalize: str = "row") -> None:
    fig_w = max(6, 0.45 * len(conf.columns) + 3)
    fig_h = max(4, 0.4 * len(conf.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_confusion(conf, ax, title, normalize)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    cols = [c for c in ("lineage_acc", "exact_acc", "coarser", "unassigned", "wrong", "hF1", "macro_f1", "ARI_assigned", "NMI_assigned",
                        "spatial_coherence", "marker_specificity") if c in summary.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(1.9 * len(cols), 4), sharey=False)
    for ax, c in zip(np.atleast_1d(axes), cols):
        vals = summary[c]
        ax.bar(range(len(vals)), vals.to_numpy(), color=["#4c72b0" if m != "reference" else "#999999" for m in summary.index])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(summary.index, rotation=70, ha="right", fontsize=8)
        ax.set_title(c, fontsize=9)
        for i, v in enumerate(vals.to_numpy()):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spatial(coords: np.ndarray, labellings: dict[str, pd.Series], path: Path, max_points: int = 150_000, seed: int = 0) -> None:
    """Side-by-side spatial maps with one shared colour per label across panels."""
    rng = np.random.default_rng(seed)
    n = coords.shape[0]
    idx = np.sort(rng.choice(n, min(n, max_points), replace=False))
    all_labels = pd.concat([s.astype(str) for s in labellings.values()])
    order = all_labels.value_counts().index.tolist()
    palette = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors
    color = {lab: palette[i % len(palette)] for i, lab in enumerate(l for l in order if l != UNASSIGNED)}
    color[UNASSIGNED] = (0.85, 0.85, 0.85)
    k = len(labellings)
    fig, axes = plt.subplots(1, k, figsize=(6 * k, 6), squeeze=False)
    for ax, (name, labs) in zip(axes[0], labellings.items()):
        labs = labs.astype(str).to_numpy()[idx]
        c = np.array([color.get(l, (0.5, 0.5, 0.5)) for l in labs])
        ax.scatter(coords[idx, 0], coords[idx, 1], c=c, s=0.3, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    top = [l for l in order if l in color][:24]
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color[l], label=l, markersize=6) for l in top]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(top)), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _md(df: pd.DataFrame, index: bool = True) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:  # tabulate missing
        return "```\n" + df.to_string(index=index) + "\n```"


def write_markdown(out: Path, dataset: str, summary: pd.DataFrame, per_class: dict[str, pd.DataFrame], tree_txt: str, notes: list[str]) -> Path:
    lines = [f"# Benchmark: {dataset}", ""]
    lines += ["## Summary", "", _md(summary.round(3)), ""]
    lines += ["Metric definitions:",
              "- `lineage_acc`: prediction equals the reference node or a descendant of it (correct at the reference's granularity).",
              "- `exact_acc`: prediction equals the reference node. `coarser`: prediction is a strict ancestor of the reference (right lineage, less specific).",
              "- `wrong`: prediction is neither on the reference's path nor below it. `unassigned`: no call. lineage_acc + coarser + unassigned + wrong = 1.",
              "- `hP/hR/hF1`: hierarchical precision/recall/F1 over ancestor-closed label sets.",
              "- `ARI/NMI_assigned`: partition agreement on assigned cells after projecting predictions onto the reference classes; `_all` treats unassigned as a class.",
              "- `spatial_coherence`: mean fraction of the k nearest spatial neighbours sharing the cell's label (assigned cells only).",
              "- `marker_specificity`: mean z of a group's own knowledge markers inside vs outside the group (cell-weighted mean over groups; not a reference-based metric).",
              "- Per-class precision/recall are hierarchical: a prediction is positive for a class when it is that node or a descendant of it, "
              "so a `pericyte` call counts for both `pericyte` and its ancestor `smooth muscle cell`. Confusion tables project each prediction onto "
              "the most specific reference class on its path (a `more_specific` column for a coarser reference row is therefore not an error).",
              ""]
    for m, df in per_class.items():
        lines += [f"## Per-class: {m}", "", _md(df.round(3), index=False), ""]
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    lines += ["## Knowledge tree", "", "```", tree_txt, "```", ""]
    p = out / "report.md"
    p.write_text("\n".join(lines))
    return p
