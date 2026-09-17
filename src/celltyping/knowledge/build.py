"""Build a tissue-specific, panel-aware cell-type hierarchy with markers.

Pipeline::

    tissue name --> UBERON id
                --> expected CL cell types   (Ubergraph part_of/located_in + marker-DB tissue columns)
                --> CL is_a DAG over those types and their ancestors --> single-parent tree
                --> attach markers from CellMarker2 / PanglaoDB / ASCT+B / user overrides
                --> filter to panel, prune nodes with too few markers, weight for sibling specificity
                --> JSON tree + coverage report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from ..utils import log, norm_gene
from .cellxgene import DEFAULT_CENSUS_VERSION, cellguide_markers, census_cell_types, census_seeds, census_tissue_general
from .markers import load_custom_markers, map_label_to_cl, markers_for_tissue
from .ontology import (
    ABSTRACT_CL,
    CL_ROOT,
    Tissue,
    clean_label,
    dag_to_tree_parents,
    expected_cell_types,
    induced_is_a_dag,
    is_other_species_term,
    load_cl,
    load_uberon,
    resolve_tissue,
    tissue_ancestors,
)
from .panel import load_panel
from .tree import CellTypeTree

# Parent choices that biologists expect but that the "deepest is_a parent" rule may not pick.
DEFAULT_PARENTS: dict[str, str] = {
    "CL:0000129": "CL:0000125",  # microglial cell -> glial cell
    "CL:0002453": "CL:0000125",  # oligodendrocyte precursor cell -> glial cell
    "CL:0000644": "CL:0000127",  # Bergmann glial cell -> astrocyte
    "CL:0012000": "CL:0000127",  # astrocyte of the forebrain -> astrocyte
    "CL:0000669": "CL:0008034",  # pericyte -> mural cell
    "CL:0000359": "CL:0008034",  # vascular associated smooth muscle cell -> mural cell
}


@dataclass
class Overrides:
    add_cell_types: list[dict] = field(default_factory=list)
    remove_cell_types: list[str] = field(default_factory=list)  # removed with their subtree
    collapse_cell_types: list[str] = field(default_factory=list)  # removed, children re-attached to grandparent
    parents: dict[str, str] = field(default_factory=dict)
    markers: dict[str, dict] = field(default_factory=dict)
    custom_markers: str | None = None

    @classmethod
    def load(cls, path: str | Path | None) -> "Overrides":
        if path is None:
            return cls()
        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            add_cell_types=d.get("add_cell_types", []) or [],
            remove_cell_types=d.get("remove_cell_types", []) or [],
            collapse_cell_types=d.get("collapse_cell_types", []) or [],
            parents={str(k): str(v) for k, v in (d.get("parents") or {}).items()},
            markers={str(k): v for k, v in (d.get("markers") or {}).items()},
            custom_markers=d.get("custom_markers"),
        )


@dataclass
class BuildParams:
    species: str = "human"
    sources: tuple[str, ...] = ("CellMarker2", "PanglaoDB", "ASCT+B", "CellGuide")
    ancestor_levels: int = 1  # also query cell types of parent tissues/systems (brain -> CNS)
    census: bool = True  # CZ CELLxGENE Census: cell types observed in the tissue are tissue evidence (seeds)
    census_version: str = DEFAULT_CENSUS_VERSION
    census_min_cells: int = 200  # a Census cell type needs this many primary cells in the tissue ...
    census_min_datasets: int = 2  # ... from this many datasets ...
    census_min_frac: float = 0.001  # ... and this fraction of the tissue's Census cells, to seed the tree on its own
    census_disease: str = "normal"  # 'normal' (healthy samples only; disease-specific types come from overrides) | 'any'
    corroborate_ancestor_hits: bool = True  # ancestor-tissue ontology hits need >=1 tissue-matched DB row
    min_db_sources: int = 2  # DB-only cell types need this many sources annotating them to the tissue
    min_markers: int = 3  # nodes with fewer *curated* panel markers are collapsed/dropped
    computational_sources: tuple[str, ...] = ("CellGuide",)  # data-derived markers: refine nodes, do not keep them alive
    max_depth: int = 6
    collapse_single_child: bool = True
    propagate_shared_to_parent: bool = True
    idf_weights: bool = False  # down-weight genes listed for many cell types (off: favours unique but noisy DB genes)


def _resolve_node_id(key: str) -> str | None:
    if key.startswith("CL:"):
        return key
    return map_label_to_cl(key)


def build_tree_skeleton(seeds: set[str], forced_parents: dict[str, str]) -> CellTypeTree:
    cl = load_cl()
    dag = induced_is_a_dag(cl, seeds)
    parents = dag_to_tree_parents(dag, prefer=seeds, forced={**DEFAULT_PARENTS, **forced_parents})
    tree = CellTypeTree(CL_ROOT, cl.label(CL_ROOT))
    # add in an order where parents exist: iterate by depth
    pending = {n for n in parents if n != CL_ROOT}
    while pending:
        progressed = False
        for n in sorted(pending):
            p = parents[n]
            if p in tree:
                tree.add(n, cl.label(n), p, seed=n in seeds)
                pending.discard(n)
                progressed = True
        if not progressed:  # cycle or missing parent -> attach to root
            for n in sorted(pending):
                tree.add(n, cl.label(n), CL_ROOT, seed=n in seeds)
            pending.clear()
    for a in list(ABSTRACT_CL):
        if a in tree and a != tree.root:
            tree.remove(a, keep_children=True)
    return tree


def attach_markers(tree: CellTypeTree, mk: pd.DataFrame, panel: set[str] | None) -> None:
    if mk.empty:
        return
    mk = mk[mk["cl_id"].isin(tree.nodes.keys())]
    # per (node, gene, source): max weight; then sources are summed by add_marker
    agg = mk.groupby(["cl_id", "gene", "source"], as_index=False)["weight"].max()
    for r in agg.itertuples(index=False):
        tree.add_marker(r.cl_id, r.gene, float(r.weight), r.source, in_panel=(panel is None or r.gene in panel))


def apply_marker_overrides(tree: CellTypeTree, ov: Overrides, panel: set[str] | None) -> None:
    for key, spec in ov.markers.items():
        nid = _resolve_node_id(key)
        if nid is None or nid not in tree:
            log.warning("override markers: node '%s' not in tree; skipped", key)
            continue
        spec = spec or {}
        if spec.get("replace"):  # discard database markers (e.g. pan-cancer rows for 'malignant cell') and use the curated list
            tree.nodes[nid].markers.clear()
            spec = {**spec, "add": list(spec["replace"]) + list(spec.get("add", []) or [])}
        for g in spec.get("add", []) or []:
            g = norm_gene(g)
            tree.add_marker(nid, g, 2.0, "override", in_panel=(panel is None or g in panel))
        for g in spec.get("remove", []) or []:
            tree.nodes[nid].markers.pop(norm_gene(g), None)


def n_curated_markers(node, computational: tuple[str, ...]) -> int:
    """Panel markers backed by at least one curated source (databases, overrides, children), i.e. not only by
    computational sources such as CellGuide, whose genes exist for almost every CL term."""
    comp = set(computational)
    return sum(1 for m in node.panel_markers.values() if set(m.sources) - comp)


def prune(tree: CellTypeTree, params: BuildParams, panel: set[str] | None) -> list[dict]:
    """Drop/collapse nodes lacking (curated) panel markers, enforce depth, collapse single-child chains."""
    dropped: list[dict] = []

    def _marker_prune() -> None:
        changed = True
        while changed:
            changed = False
            for nid in sorted(tree.nodes, key=lambda i: -tree.depth(i)):  # deepest first
                if nid == tree.root or nid not in tree:
                    continue
                n = tree.nodes[nid]
                if n_curated_markers(n, params.computational_sources) < params.min_markers:
                    if n.children:
                        dropped.append({"id": nid, "label": n.label, "reason": f"<{params.min_markers} curated panel markers (children kept)", "n_markers_total": len(n.markers)})
                        tree.remove(nid, keep_children=True)
                    else:
                        dropped.append({"id": nid, "label": n.label, "reason": f"<{params.min_markers} curated panel markers", "n_markers_total": len(n.markers)})
                        tree.remove(nid, keep_children=False)
                    changed = True

    def _depth_prune() -> None:
        """Bring deep nodes within ``max_depth`` by collapsing the weakest non-seed ancestor; drop only as last resort."""
        changed = True
        while changed:
            changed = False
            for nid in sorted(tree.nodes, key=lambda i: -tree.depth(i)):
                if nid not in tree or tree.depth(nid) <= params.max_depth:
                    continue
                anc = [a for a in tree.path(nid)[1:-1] if not tree.nodes[a].seed]
                if anc:
                    victim = min(anc, key=lambda a: len(tree.nodes[a].panel_markers))
                    dropped.append({"id": victim, "label": tree.nodes[victim].label, "reason": f"collapsed to respect max_depth={params.max_depth}", "n_markers_total": len(tree.nodes[victim].markers)})
                    tree.remove(victim, keep_children=True)
                else:
                    dropped.append({"id": nid, "label": tree.nodes[nid].label, "reason": f"depth>{params.max_depth}", "n_markers_total": len(tree.nodes[nid].markers)})
                    tree.remove(nid, keep_children=False)
                changed = True
                break  # depths changed; recompute ordering

    def _structural_leaf_prune() -> None:
        """Non-seed nodes exist only as ancestors of seeds; if the tree-ification attached their
        descendants elsewhere (multiple is_a parents) they end up as childless dead ends -> drop."""
        changed = True
        while changed:
            changed = False
            for nid in list(tree.nodes):
                if nid == tree.root or nid not in tree:
                    continue
                n = tree.nodes[nid]
                if not n.seed and not n.children:
                    dropped.append({"id": nid, "label": n.label, "reason": "non-seed node without descendants", "n_markers_total": len(n.markers)})
                    tree.remove(nid, keep_children=False)
                    changed = True

    _structural_leaf_prune()
    _marker_prune()
    _depth_prune()
    _marker_prune()
    _structural_leaf_prune()
    if params.collapse_single_child:
        changed = True
        while changed:
            changed = False
            for nid in list(tree.nodes):
                if nid == tree.root or nid not in tree:
                    continue
                n = tree.nodes[nid]
                if len(n.children) == 1 and not n.seed:
                    dropped.append({"id": nid, "label": n.label, "reason": "single-child non-seed node collapsed", "n_markers_total": len(n.markers)})
                    tree.remove(nid, keep_children=True)
                    changed = True
    return dropped


def _sibling_gene_counts(tree: CellTypeTree, pid: str) -> dict[str, int]:
    count: dict[str, int] = {}
    for k in tree.nodes[pid].children:
        for g in tree.nodes[k].markers:
            count[g] = count.get(g, 0) + 1
    return count


def propagate_shared(tree: CellTypeTree) -> None:
    """Lift genes shared by *all* children of a node to that node (bottom-up)."""
    for pid in sorted(tree.internal(), key=lambda i: -tree.depth(i)):
        kids = tree.nodes[pid].children
        if len(kids) < 2 or pid == tree.root:
            continue
        for g, c in _sibling_gene_counts(tree, pid).items():
            if c == len(kids) and g not in tree.nodes[pid].markers:
                ms = [tree.nodes[k].markers[g] for k in kids]
                tree.add_marker(pid, g, sum(m.weight for m in ms) / len(ms), "shared-by-children", in_panel=all(m.in_panel for m in ms))


def reweight_siblings(tree: CellTypeTree) -> None:
    """Down-weight a gene in a child by the number of siblings that also list it (specificity)."""
    for pid in tree.internal():
        kids = tree.nodes[pid].children
        if len(kids) < 2:
            continue
        for g, c in _sibling_gene_counts(tree, pid).items():
            if c > 1:
                for k in kids:
                    m = tree.nodes[k].markers.get(g)
                    if m is not None:
                        m.weight = m.weight / c


def apply_specificity_weights(tree: CellTypeTree) -> None:
    """IDF-style down-weighting: a gene listed for many cell types in the tree is a weak marker for any of them.

    ``w *= 1 / (1 + log2(n_nodes_listing_gene))`` -- unique markers keep full weight, a gene listed
    by 8 cell types is worth a quarter.
    """
    import math

    count: dict[str, int] = {}
    for n in tree.nodes.values():
        for g in n.markers:
            count[g] = count.get(g, 0) + 1
    for n in tree.nodes.values():
        for g, m in n.markers.items():
            if count[g] > 1:
                m.weight = m.weight / (1.0 + math.log2(count[g]))


def coverage_report(tree: CellTypeTree, dropped: list[dict], computational: tuple[str, ...] = ("CellGuide",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for n in tree.iter_bfs():
        pm = sorted(n.panel_markers.values(), key=lambda m: -m.weight)
        rows.append({
            "id": n.id,
            "label": n.label,
            "depth": tree.depth(n.id),
            "parent": n.parent,
            "parent_label": tree.nodes[n.parent].label if n.parent else None,
            "n_children": len(n.children),
            "seed": n.seed,
            "n_markers_total": len(n.markers),
            "n_markers_in_panel": len(pm),
            "n_curated_in_panel": n_curated_markers(n, computational),
            "census_n_cells": (n.census or {}).get("n_cells"),
            "census_n_datasets": (n.census or {}).get("n_datasets"),
            "sources": ";".join(sorted({s for m in pm for s in m.sources})),
            "top_panel_markers": ", ".join(m.gene for m in pm[:10]),
        })
    return pd.DataFrame(rows), pd.DataFrame(dropped, columns=["id", "label", "reason", "n_markers_total"])


def build_knowledge(tissue: str, panel=None, params: BuildParams | None = None, overrides: str | Path | Overrides | None = None,
                    out_dir: str | Path | None = None, force: bool = False) -> tuple[CellTypeTree, pd.DataFrame, pd.DataFrame]:
    """End-to-end knowledge build. Returns ``(tree, coverage, dropped)`` and writes them to ``out_dir`` if given."""
    params = params or BuildParams()
    ov = overrides if isinstance(overrides, Overrides) else Overrides.load(overrides)
    t: Tissue = resolve_tissue(tissue)
    log.info("tissue: %s (%s)", t.label, t.id)
    panel_set = set(load_panel(panel)) if panel is not None else None
    if panel_set is not None:
        log.info("panel: %d genes", len(panel_set))

    # 1. expected cell types from the ontology (tissue + its part_of ancestors, e.g. brain -> CNS)
    onto = expected_cell_types(t, params.species, params.ancestor_levels)
    if params.ancestor_levels:
        log.info("ontology scope: %s", ", ".join([t.label] + [load_uberon().label(a) for a in tissue_ancestors(t.id, params.ancestor_levels)]))

    # 2. markers (all sources); DB rows count as tissue evidence only if their tissue is within the
    #    target tissue or contains it (not for sibling tissues of a shared parent system)
    mk = markers_for_tissue(t, params.species, params.sources, tissue_ids=[t.id], force=force)
    if ov.custom_markers:
        mk = pd.concat([mk, load_custom_markers(ov.custom_markers, params.species).assign(tissue_match=True)], ignore_index=True)
    matched = mk[mk["tissue_match"]] if not mk.empty else mk
    db_support = matched.groupby("cl_id")["source"].nunique() if not matched.empty else pd.Series(dtype=int)

    # 2b. CZ CELLxGENE Census: cell types actually observed in the tissue (expert annotations, CL-normalised)
    census_df = pd.DataFrame()
    tissue_general = None
    if params.census or "CellGuide" in params.sources:
        tissue_general = census_tissue_general(t)
        if tissue_general is None:
            log.warning("no Census tissue_general category contains %s; Census/CellGuide restricted to organism-wide entries", t.label)
    if params.census and tissue_general is not None:
        census_all = census_cell_types(tissue_general[1], params.species, params.census_version, params.census_disease, force=force)
        census_df = census_seeds(census_all, params.species, params.census_min_cells, params.census_min_datasets, params.census_min_frac)
        log.info("Census (%s, tissue_general=%s, disease=%s): %d cell types observed, %d pass >=%d cells, >=%d datasets, >=%.2g of cells",
                 params.census_version, tissue_general[1], params.census_disease, len(census_all), len(census_df),
                 params.census_min_cells, params.census_min_datasets, params.census_min_frac)
    cx_seeds = set(census_df["cl_id"]) if not census_df.empty else set()
    # a Census observation counts as one corroborating source next to the marker databases
    support_ids = set(db_support.index) | cx_seeds
    db_support = db_support.add(pd.Series(1, index=sorted(cx_seeds)), fill_value=0).astype(int) if cx_seeds else db_support
    db_seeds = set(db_support[db_support >= params.min_db_sources].index)
    if not onto.empty:
        direct = set(onto.loc[onto["tissue_id"] == t.id, "cl_id"])
        via_ancestor = set(onto.loc[onto["tissue_id"] != t.id, "cl_id"])
        if params.corroborate_ancestor_hits:
            via_ancestor &= support_ids
        onto_seeds = direct | via_ancestor
    else:
        onto_seeds = set()
    seeds = onto_seeds | db_seeds | cx_seeds
    # coarse Census annotations ('neural cell', 'hematopoietic cell') that are is_a ancestors of other seeds would add
    # a hierarchy level without being a typing target: keep them as structural nodes only
    cx_only = cx_seeds - onto_seeds - db_seeds
    if cx_only:
        _cl = load_cl()
        anc_of_others = set()
        for sd in seeds:
            anc_of_others |= _cl.ancestors(sd, ("is_a",)) - {sd}
        demoted = cx_only & anc_of_others
        if demoted:
            log.info("Census-only seeds demoted to structural nodes (ancestors of other seeds): %s", ", ".join(sorted(_cl.label(d) for d in demoted)))
            seeds -= demoted
            cx_seeds -= demoted
    for spec in ov.add_cell_types:
        nid = _resolve_node_id(spec["id"] if isinstance(spec, dict) else spec)
        if nid:
            seeds.add(nid)
    seeds -= set(ov.remove_cell_types)
    log.info("expected cell types: %d from ontology (%d direct), %d from >=%d marker DBs/Census, %d from Census, %d total",
             len(onto_seeds), len(direct) if not onto.empty else 0, len(db_seeds), params.min_db_sources, len(cx_seeds), len(seeds))

    # 3. tree skeleton
    forced = {**ov.parents}
    for spec in ov.add_cell_types:
        if isinstance(spec, dict) and spec.get("parent"):
            forced[_resolve_node_id(spec["id"])] = _resolve_node_id(spec["parent"])
    tree = build_tree_skeleton(seeds, forced)
    cl = load_cl()
    for spec in ov.add_cell_types:  # non-CL custom nodes
        if isinstance(spec, dict) and not str(spec["id"]).startswith("CL:") and _resolve_node_id(spec["id"]) is None:
            tree.add(spec["id"], spec.get("label", spec["id"]), _resolve_node_id(spec.get("parent", CL_ROOT)) or CL_ROOT, seed=True)
    for rid in ov.remove_cell_types:
        if rid in tree:
            tree.remove(rid, keep_children=False)
    for rid in ov.collapse_cell_types:
        if rid in tree and rid != tree.root:
            tree.remove(rid, keep_children=True)
    # provisional CL terms of the other species (e.g. '... (Mmus)') are not classification targets
    for nid in list(tree.nodes):
        if nid in tree and nid != tree.root and is_other_species_term(cl.label(nid) if nid in cl else "", params.species):
            tree.remove(nid, keep_children=True)
    for n in tree.nodes.values():
        if n.id in onto_seeds:
            n.origin.append("ontology")
        if n.id in db_seeds:
            n.origin.append("marker-db")
        if n.id in cx_seeds:
            n.origin.append("census")
        n.label = clean_label(cl.label(n.id)) if n.id in cl else n.label

    # 4. markers (CellGuide needs the node list, so it is fetched after the skeleton)
    attach_markers(tree, mk, panel_set)
    if "CellGuide" in params.sources:
        # marker evidence, not tissue evidence: seeds only, so that structural intermediates (CellGuide
        # has genes for almost every CL term) do not survive pruning just because CellGuide lists them
        targets = [n.id for n in tree.nodes.values() if n.seed]
        cg = cellguide_markers(targets, params.species, tissue_general, force=force)
        attach_markers(tree, cg, panel_set)
    if not census_df.empty:
        census_by_id = census_df.set_index("cl_id")
        for n in tree.nodes.values():
            if n.id in census_by_id.index:
                n.census = {"n_cells": int(census_by_id.at[n.id, "n_cells"]), "n_datasets": int(census_by_id.at[n.id, "n_datasets"])}
    apply_marker_overrides(tree, ov, panel_set)
    if params.propagate_shared_to_parent:
        propagate_shared(tree)  # lift shared genes so informative parents survive pruning

    # 5. prune to the panel, then weight for sibling specificity
    dropped = prune(tree, params, panel_set)
    if params.propagate_shared_to_parent:
        propagate_shared(tree)
    reweight_siblings(tree)
    if params.idf_weights:
        apply_specificity_weights(tree)

    tree.meta = {
        "tissue": {"id": t.id, "label": t.label},
        "census_tissue_general": tissue_general[1] if tissue_general else None,
        "species": params.species,
        "sources": list(params.sources),
        "panel_size": len(panel_set) if panel_set is not None else None,
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.__dict__.items()},
        "n_nodes": len(tree),
        "n_leaves": len(tree.leaves()),
    }
    cov, drop_df = coverage_report(tree, dropped, params.computational_sources)
    log.info("tree: %d nodes, %d leaves, max depth %d; %d nodes dropped", len(tree), len(tree.leaves()), tree.max_depth(), len(drop_df))
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = t.slug
        tree.save(out_dir / f"{stem}_tree.json")
        cov.to_csv(out_dir / f"{stem}_coverage.csv", index=False)
        drop_df.to_csv(out_dir / f"{stem}_dropped.csv", index=False)
        (out_dir / f"{stem}_tree.txt").write_text(tree.render())
        log.info("wrote %s/%s_{tree.json,coverage.csv,dropped.csv,tree.txt}", out_dir, stem)
    return tree, cov, drop_df
