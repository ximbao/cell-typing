"""A simple rooted cell-type tree with per-node marker sets.

The tree is the central data structure shared by the knowledge builder and the
annotators. Nodes are keyed by Cell Ontology ids (``CL:0000127``) or, for
user-defined types, any unique string id.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

ROOT_ID = "CL:0000000"


@dataclass
class Marker:
    gene: str
    weight: float = 1.0
    sources: list[str] = field(default_factory=list)
    in_panel: bool = True

    def to_dict(self) -> dict:
        return {"gene": self.gene, "weight": round(float(self.weight), 4), "sources": sorted(set(self.sources)), "in_panel": self.in_panel}


@dataclass
class Node:
    id: str
    label: str
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    markers: dict[str, Marker] = field(default_factory=dict)
    seed: bool = False  # directly evidenced for this tissue (ontology part_of or DB tissue column)
    origin: list[str] = field(default_factory=list)  # provenance notes
    census: dict | None = None  # {"n_cells", "n_datasets"} observed for this type in the tissue (CZ CELLxGENE Census)

    @property
    def panel_markers(self) -> dict[str, Marker]:
        return {g: m for g, m in self.markers.items() if m.in_panel}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "parent": self.parent,
            "children": list(self.children),
            "seed": self.seed,
            "origin": sorted(set(self.origin)),
            "census": self.census,
            "n_markers_total": len(self.markers),
            "n_markers_in_panel": len(self.panel_markers),
            "markers": [m.to_dict() for m in sorted(self.markers.values(), key=lambda m: (-m.weight, m.gene))],
        }


class CellTypeTree:
    """Rooted tree; ``root`` is always present."""

    def __init__(self, root_id: str = ROOT_ID, root_label: str = "cell"):
        self.root = root_id
        self.nodes: dict[str, Node] = {root_id: Node(root_id, root_label)}
        self.meta: dict = {}

    # ------------------------------------------------------------------ basics
    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def add(self, node_id: str, label: str, parent: str | None = None, **kw) -> Node:
        if node_id in self.nodes:
            n = self.nodes[node_id]
            if parent is not None and parent != n.parent:
                self.reparent(node_id, parent)
            return n
        parent = parent or self.root
        if parent not in self.nodes:
            raise KeyError(f"parent {parent} not in tree")
        n = Node(node_id, label, parent=parent, **kw)
        self.nodes[node_id] = n
        self.nodes[parent].children.append(node_id)
        return n

    def reparent(self, node_id: str, new_parent: str) -> None:
        n = self.nodes[node_id]
        if new_parent == node_id or new_parent in self.descendants(node_id):
            raise ValueError(f"cannot reparent {node_id} under its own descendant {new_parent}")
        if n.parent is not None:
            self.nodes[n.parent].children.remove(node_id)
        n.parent = new_parent
        self.nodes[new_parent].children.append(node_id)

    def remove(self, node_id: str, keep_children: bool = True) -> None:
        """Remove a node. Children are re-attached to its parent unless ``keep_children`` is False."""
        if node_id == self.root:
            raise ValueError("cannot remove root")
        n = self.nodes[node_id]
        if keep_children:
            for c in list(n.children):
                self.reparent(c, n.parent)
        else:
            for c in list(n.children):
                self.remove(c, keep_children=False)
        self.nodes[n.parent].children.remove(node_id)
        del self.nodes[node_id]

    # --------------------------------------------------------------- traversal
    def depth(self, node_id: str) -> int:
        d, cur = 0, self.nodes[node_id]
        while cur.parent is not None:
            d += 1
            cur = self.nodes[cur.parent]
        return d

    def path(self, node_id: str) -> list[str]:
        out, cur = [], self.nodes[node_id]
        while cur is not None:
            out.append(cur.id)
            cur = self.nodes[cur.parent] if cur.parent is not None else None
        return out[::-1]

    def descendants(self, node_id: str) -> set[str]:
        out, stack = set(), list(self.nodes[node_id].children)
        while stack:
            c = stack.pop()
            out.add(c)
            stack.extend(self.nodes[c].children)
        return out

    def leaves(self) -> list[str]:
        return [i for i, n in self.nodes.items() if not n.children]

    def internal(self) -> list[str]:
        return [i for i, n in self.nodes.items() if n.children]

    def iter_bfs(self) -> Iterator[Node]:
        queue = [self.root]
        while queue:
            cur = queue.pop(0)
            yield self.nodes[cur]
            queue.extend(self.nodes[cur].children)

    def max_depth(self) -> int:
        return max(self.depth(i) for i in self.nodes)

    # ----------------------------------------------------------------- markers
    def add_marker(self, node_id: str, gene: str, weight: float = 1.0, source: str | None = None, in_panel: bool = True) -> None:
        n = self.nodes[node_id]
        m = n.markers.get(gene)
        if m is None:
            n.markers[gene] = Marker(gene, weight, [source] if source else [], in_panel)
        else:
            m.weight = max(m.weight, weight) if source in m.sources else m.weight + weight
            if source and source not in m.sources:
                m.sources.append(source)
            m.in_panel = m.in_panel or in_panel

    def marker_sets(self, node_ids: list[str] | None = None, panel_only: bool = True, top_k: int | None = 50) -> dict[str, dict[str, float]]:
        """``{node_id: {gene: weight}}`` for the requested nodes (default: all non-root).

        ``top_k`` keeps only the highest-weighted markers per node (None = all) so that
        database entries with hundreds of genes do not dilute the signature.
        """
        ids = node_ids if node_ids is not None else [i for i in self.nodes if i != self.root]
        out = {}
        for i in ids:
            ms = self.nodes[i].panel_markers if panel_only else self.nodes[i].markers
            items = sorted(ms.values(), key=lambda m: (-m.weight, m.gene))
            if top_k is not None:
                items = items[:top_k]
            out[i] = {m.gene: m.weight for m in items}
        return out

    def subtree_marker_sets(self, node_ids: list[str], panel_only: bool = True, top_k: int | None = 50,
                            sibling_specific: bool = True) -> dict[str, dict[str, float]]:
        """Signature of each node's *subtree* (own markers plus all descendants').

        A gene's weight is the maximum over the subtree plus a small bonus for recurring in
        several descendants. With ``sibling_specific`` genes present in several of the requested
        (sibling) sets are divided by that count, so lineage decisions rely on lineage-specific
        genes. Used for internal nodes, whose own database markers are sparse and noisy.
        """
        raw: dict[str, dict[str, float]] = {}
        for nid in node_ids:
            acc: dict[str, list[float]] = {}
            # structural (non-seed) internal nodes, e.g. 'epithelial cell', carry pan-tissue database rows
            # that would swamp the subtree; the descendants define the lineage signature instead
            for member in self.informative_members(nid):
                ms = self.nodes[member].panel_markers if panel_only else self.nodes[member].markers
                for g, mk in ms.items():
                    acc.setdefault(g, []).append(mk.weight)
            raw[nid] = {g: max(ws) + 0.25 * math.log1p(len(ws) - 1) for g, ws in acc.items()}
        if sibling_specific and len(raw) > 1:
            count: dict[str, int] = {}
            for gw in raw.values():
                for g in gw:
                    count[g] = count.get(g, 0) + 1
            for gw in raw.values():
                for g in gw:
                    if count[g] > 1:
                        gw[g] /= count[g]
        out = {}
        for nid, gw in raw.items():
            items = sorted(gw.items(), key=lambda kv: (-kv[1], kv[0]))
            out[nid] = dict(items[:top_k] if top_k is not None else items)
        return out

    def informative_members(self, node_id: str) -> list[str]:
        """Node plus descendants, without structural (non-seed) internal nodes; falls back to all members."""
        members = [node_id, *self.descendants(node_id)]
        return [m for m in members if not (self.nodes[m].children and not self.nodes[m].seed)] or members

    def subtree_member_sets(self, node_ids: list[str], panel_only: bool = True, top_k: int | None = 50,
                            sibling_specific: bool = True) -> dict[str, dict[str, dict[str, float]]]:
        """Per requested node, the *separate* marker sets of its informative subtree members.

        Alternative to :meth:`subtree_marker_sets` for heterogeneous subtrees (e.g. 'epithelial cell'
        holding tumour, tubal, urothelial and mesothelial cells): the caller scores each member set and
        takes the maximum per cell, so a cell that matches any one member well is recognised, instead of
        being diluted by the union. Sibling specificity divides a gene's weight by the number of
        requested (sibling) nodes whose members list it.
        """
        raw: dict[str, dict[str, dict[str, float]]] = {}
        for nid in node_ids:
            raw[nid] = {}
            for member in self.informative_members(nid):
                ms = self.nodes[member].panel_markers if panel_only else self.nodes[member].markers
                if ms:
                    raw[nid][member] = {g: m.weight for g, m in ms.items()}
        if sibling_specific and len(raw) > 1:
            count: dict[str, int] = {}
            for sets in raw.values():
                for g in {g for gw in sets.values() for g in gw}:
                    count[g] = count.get(g, 0) + 1
            for sets in raw.values():
                for gw in sets.values():
                    for g in gw:
                        if count[g] > 1:
                            gw[g] /= count[g]
        out: dict[str, dict[str, dict[str, float]]] = {}
        for nid, sets in raw.items():
            out[nid] = {}
            for member, gw in sets.items():
                items = sorted(gw.items(), key=lambda kv: (-kv[1], kv[0]))
                out[nid][member] = dict(items[:top_k] if top_k is not None else items)
        return out

    # ------------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        return {"root": self.root, "meta": self.meta, "nodes": [n.to_dict() for n in self.iter_bfs()]}

    @classmethod
    def from_dict(cls, d: dict) -> "CellTypeTree":
        nodes = d["nodes"]
        root = d.get("root", ROOT_ID)
        root_label = next((n["label"] for n in nodes if n["id"] == root), "cell")
        t = cls(root, root_label)
        t.meta = d.get("meta", {})
        for nd in nodes:  # BFS order guarantees parents come first
            if nd["id"] == root:
                node = t.nodes[root]
            else:
                node = t.add(nd["id"], nd["label"], nd["parent"])
            node.seed = nd.get("seed", False)
            node.origin = list(nd.get("origin", []))
            node.census = nd.get("census")
            for m in nd.get("markers", []):
                node.markers[m["gene"]] = Marker(m["gene"], m.get("weight", 1.0), list(m.get("sources", [])), m.get("in_panel", True))
        return t

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CellTypeTree":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------ display
    def render(self, show_markers: int = 6, panel_only: bool = True) -> str:
        lines = []

        def _walk(node_id: str, prefix: str, last: bool):
            n = self.nodes[node_id]
            ms = n.panel_markers if panel_only else n.markers
            top = sorted(ms.values(), key=lambda m: -m.weight)[:show_markers]
            mk = ", ".join(m.gene for m in top)
            more = f" (+{len(ms) - len(top)})" if len(ms) > len(top) else ""
            branch = "" if node_id == self.root else ("└─ " if last else "├─ ")
            lines.append(f"{prefix}{branch}{n.label} [{n.id}] n={len(ms)}: {mk}{more}")
            ext = "" if node_id == self.root else ("   " if last else "│  ")
            kids = n.children
            for i, c in enumerate(kids):
                _walk(c, prefix + ext, i == len(kids) - 1)

        _walk(self.root, "", True)
        return "\n".join(lines)
