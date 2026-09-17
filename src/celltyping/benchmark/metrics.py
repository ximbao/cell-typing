"""Benchmark metrics: agreement with a reference, hierarchical P/R/F, marker specificity, spatial coherence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..methods.scoring import UNASSIGNED
from ..knowledge.tree import CellTypeTree
from ..preprocess import zscore_matrix
from .harmonize import (
    ANCESTOR,
    DESCENDANT,
    EXACT,
    UNASSIGNED_REL,
    WRONG,
    ancestor_sets,
    is_unassigned,
    project_to_reference_classes,
    relation_table,
)


# ------------------------------------------------------------------------------------- agreement
def hierarchical_prf(pred: pd.Series, ref: pd.Series, tree: CellTypeTree) -> dict[str, float]:
    """Hierarchical precision / recall / F1 (Kiritchenko et al.): overlap of ancestor-closed label sets.

    Computed over cells with a reference; unassigned predictions contribute an empty set (recall 0).
    """
    anc = ancestor_sets(tree)
    inter = pred_sz = ref_sz = 0
    for p, r in zip(pred.to_numpy(), ref.to_numpy()):
        rs = anc[r] | {r}
        ps = set() if is_unassigned(p, tree) else anc[p] | {p}
        inter += len(ps & rs)
        pred_sz += len(ps)
        ref_sz += len(rs)
    hp = inter / pred_sz if pred_sz else 0.0
    hr = inter / ref_sz if ref_sz else 0.0
    hf = 2 * hp * hr / (hp + hr) if hp + hr else 0.0
    return {"hP": hp, "hR": hr, "hF1": hf}


def agreement(pred: pd.Series, ref: pd.Series, tree: CellTypeTree) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Summary dict, per-reference-class table and confusion matrix (reference x projected prediction).

    Only cells with a reference label are evaluated. ``lineage_acc`` counts a prediction as correct
    when it is the reference node or one of its descendants (i.e. right at the reference's granularity);
    ``coarser`` predictions (a strict ancestor of the reference) are reported separately, as are
    unassigned cells, so that ``lineage_acc + coarser + unassigned + wrong == 1``.
    """
    m = ref.notna().to_numpy()
    pred, ref = pred[m].astype(object), ref[m].astype(object)
    rel = relation_table(pred, ref, tree)
    n = len(rel)
    summ = {
        "n_eval": n,
        "unassigned": float((rel == UNASSIGNED_REL).mean()),
        "exact_acc": float((rel == EXACT).mean()),
        "lineage_acc": float(rel.isin([EXACT, DESCENDANT]).mean()),
        "coarser": float((rel == ANCESTOR).mean()),
        "wrong": float((rel == WRONG).mean()),
    }
    assigned = rel != UNASSIGNED_REL
    summ["lineage_acc_of_assigned"] = float(rel[assigned].isin([EXACT, DESCENDANT]).mean()) if assigned.any() else np.nan
    summ.update(hierarchical_prf(pred, ref, tree))

    ref_classes = set(ref.unique())
    proj = project_to_reference_classes(pred, ref_classes, tree)
    # ARI/NMI: on assigned cells, and with 'unassigned' as its own class
    pa = proj[assigned].to_numpy(dtype=object)
    ra = ref[assigned].to_numpy(dtype=object)
    summ["ARI_assigned"] = float(adjusted_rand_score(ra, pa)) if assigned.sum() > 1 else np.nan
    summ["NMI_assigned"] = float(normalized_mutual_info_score(ra, pa)) if assigned.sum() > 1 else np.nan
    summ["ARI_all"] = float(adjusted_rand_score(ref.to_numpy(dtype=object), proj.to_numpy(dtype=object)))
    summ["NMI_all"] = float(normalized_mutual_info_score(ref.to_numpy(dtype=object), proj.to_numpy(dtype=object)))

    # per-class precision/recall, hierarchical: a prediction is positive for class c when it is c or a
    # descendant of c (so a 'pericyte' call is positive for both 'pericyte' and its ancestor 'smooth muscle cell')
    anc = ancestor_sets(tree)
    pred_arr = pred.to_numpy(dtype=object)
    rows = []
    for c in sorted(ref_classes, key=lambda x: tree.nodes[x].label):
        is_ref = (ref == c).to_numpy()
        is_pred = np.array([p == c or (not is_unassigned(p, tree) and c in anc.get(p, set())) for p in pred_arr])
        tp = int((is_ref & is_pred).sum())
        fp = int((~is_ref & is_pred).sum())
        fn = int((is_ref & ~is_pred).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        # a class that is never (correctly) predicted has F1 = 0, not NaN, so macro-F1 penalises missing classes
        f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if tp + fn else np.nan)
        r_c = rel[is_ref]
        rows.append({"ref_id": c, "ref_label": tree.nodes[c].label, "n_ref": int(is_ref.sum()), "n_pred": int(is_pred.sum()),
                     "precision": prec, "recall": rec, "f1": f1,
                     "unassigned": float((r_c == UNASSIGNED_REL).mean()), "coarser": float((r_c == ANCESTOR).mean()),
                     "wrong": float((r_c == WRONG).mean())})
    per_class = pd.DataFrame(rows)
    summ["macro_f1"] = float(per_class["f1"].mean(skipna=True))
    summ["weighted_f1"] = float(np.nansum(per_class["f1"] * per_class["n_ref"]) / per_class["n_ref"].sum())

    lab = lambda x: tree.nodes[x].label if x in tree else str(x)  # noqa: E731
    conf = pd.crosstab(ref.map(lab).rename("reference"), proj.map(lab).rename("predicted"))
    return summ, per_class, conf


# ------------------------------------------------------------------------------------- marker specificity
def marker_specificity(adata, labels: pd.Series, tree: CellTypeTree, top_k: int = 30, min_cells: int = 50, layer: str | None = None) -> dict[str, float]:
    """How well a labelling separates each type's own markers: mean z of the type's markers in its cells minus in all other cells.

    Uses the same knowledge markers for every method (subtree signatures for internal nodes), so it measures
    marker coherence of the groups, not agreement with the reference. Reported as the cell-weighted mean.
    """
    labs = labels.astype(object)
    ok = labs.map(lambda p: not is_unassigned(p, tree) and p in tree).to_numpy()
    counts = labs[ok].value_counts()
    counts = counts[counts >= min_cells]
    if counts.empty:
        return {"marker_specificity": np.nan, "marker_specificity_n_types": 0}
    genes_available = set(adata.var_names)
    vals, weights = [], []
    for nid, n in counts.items():
        sets = tree.marker_sets([nid], top_k=top_k) if not tree.nodes[nid].children else tree.subtree_marker_sets([nid], top_k=top_k, sibling_specific=False)
        genes = [g for g in sets.get(nid, {}) if g in genes_available]
        if len(genes) < 3:
            continue
        Z = zscore_matrix(adata, genes, layer=layer)
        inside = (labs == nid).to_numpy()
        spec = float(Z[inside].mean() - Z[~inside].mean()) if (~inside).any() else np.nan
        vals.append(spec)
        weights.append(n)
    if not vals:
        return {"marker_specificity": np.nan, "marker_specificity_n_types": 0}
    return {"marker_specificity": float(np.average(vals, weights=weights)), "marker_specificity_n_types": len(vals)}


# ------------------------------------------------------------------------------------- spatial coherence
def spatial_coherence(coords: np.ndarray, labels: pd.Series, tree: CellTypeTree, k: int = 10) -> dict[str, float]:
    """Mean fraction of a cell's k nearest spatial neighbours (among assigned cells) that share its label."""
    from sklearn.neighbors import NearestNeighbors

    labs = labels.astype(object).to_numpy()
    ok = np.array([not is_unassigned(p, tree) for p in labs])
    if ok.sum() <= k:
        return {"spatial_coherence": np.nan}
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(coords[ok])
    idx = nn.kneighbors(coords[ok], return_distance=False)[:, 1:]
    L = labs[ok]
    same = (L[idx] == L[:, None]).mean(axis=1)
    return {"spatial_coherence": float(same.mean())}


def label_entropy(labels: pd.Series, tree: CellTypeTree) -> dict[str, float]:
    labs = labels.astype(object)
    ok = labs.map(lambda p: not is_unassigned(p, tree)).to_numpy()
    p = labs[ok].value_counts(normalize=True).to_numpy()
    return {"n_labels": int(len(p)), "label_entropy_bits": float(-(p * np.log2(p)).sum()) if len(p) else np.nan}
