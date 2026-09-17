"""Map a reference (expert) annotation onto the knowledge tree and relate predictions to it.

Reference labels are free text; the config maps them to Cell Ontology ids. A mapped id that is
not a node of the tree is lifted to its closest ``is_a`` ancestor that is (using the Cell Ontology),
so the comparison is always between two nodes of the same tree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..methods.scoring import UNASSIGNED
from ..knowledge.tree import CellTypeTree
from ..utils import log

# how a prediction relates to the reference node
EXACT, DESCENDANT, ANCESTOR, WRONG, UNASSIGNED_REL = "exact", "more_specific", "coarser", "wrong", "unassigned"


def lift_to_tree(cl_id: str, tree: CellTypeTree) -> str | None:
    """Return ``cl_id`` if it is in the tree, else its nearest Cell Ontology ancestor that is (root excluded)."""
    if cl_id in tree:
        return cl_id
    try:
        from ..knowledge.ontology import load_cl

        cl = load_cl()
    except Exception:  # pragma: no cover
        return None
    frontier, seen = [cl_id], {cl_id}
    while frontier:
        nxt = []
        for n in frontier:
            for p in (cl.parents(n) if n in cl else []):
                if p in tree and p != tree.root:
                    return p
                if p not in seen:
                    seen.add(p)
                    nxt.append(p)
        frontier = nxt
    return None


def reference_ids(labels: pd.Series, mapping: dict[str, str], tree: CellTypeTree, ignore: list[str] | None = None) -> pd.Series:
    """Map reference labels to tree node ids. Ignored / unmapped labels become ``None``."""
    ignore = set(ignore or [])
    resolved: dict[str, str | None] = {}
    for lab, target in mapping.items():
        lifted = lift_to_tree(target, tree)
        if lifted is None:
            log.warning("reference label %r -> %s is not in the tree and has no tree ancestor; excluded", lab, target)
        elif lifted != target:
            log.info("reference label %r -> %s lifted to tree node %s (%s)", lab, target, lifted, tree.nodes[lifted].label)
        resolved[lab] = lifted
    labs = labels.astype(str)
    unmapped = sorted(set(labs.unique()) - set(mapping) - ignore)
    if unmapped:
        log.warning("reference labels without mapping (excluded): %s", unmapped)
    out = labs.map(lambda l: None if l in ignore else resolved.get(l))
    return out.astype(object).where(out.notna(), None)


def is_unassigned(pred: str, tree: CellTypeTree) -> bool:
    """Root, 'unassigned', missing, or any id that is not a tree node (e.g. 'low_quality')."""
    if pred is None or pred == UNASSIGNED or pred == tree.root or (isinstance(pred, float) and np.isnan(pred)):
        return True
    return pred not in tree.nodes


def relation(pred: str, ref: str, tree: CellTypeTree, ancestors: dict[str, set[str]]) -> str:
    if is_unassigned(pred, tree):
        return UNASSIGNED_REL
    if pred == ref:
        return EXACT
    if ref in ancestors.get(pred, set()):
        return DESCENDANT
    if pred in ancestors.get(ref, set()):
        return ANCESTOR
    return WRONG


def ancestor_sets(tree: CellTypeTree) -> dict[str, set[str]]:
    """Node -> set of its strict ancestors excluding the root."""
    return {nid: set(tree.path(nid)[1:-1]) for nid in tree.nodes}


def project_to_reference_classes(pred: pd.Series, ref_classes: set[str], tree: CellTypeTree) -> pd.Series:
    """Collapse predictions onto the reference class set for confusion tables.

    A prediction inside the subtree of a reference class becomes that class; a prediction that is
    a strict ancestor of reference classes is kept under its own (coarser) label; anything else is kept as is.
    """
    cache: dict[str, str] = {}

    def _proj(p: str) -> str:
        if is_unassigned(p, tree):
            return UNASSIGNED
        if p in cache:
            return cache[p]
        out = p
        if p not in ref_classes:
            for a in tree.path(p)[::-1]:
                if a in ref_classes:
                    out = a
                    break
        cache[p] = out
        return out

    return pred.map(_proj)


def relation_table(pred: pd.Series, ref: pd.Series, tree: CellTypeTree) -> pd.Series:
    anc = ancestor_sets(tree)
    p = pred.astype(object).to_numpy()
    r = ref.astype(object).to_numpy()
    return pd.Series([relation(pi, ri, tree, anc) for pi, ri in zip(p, r)], index=pred.index)
