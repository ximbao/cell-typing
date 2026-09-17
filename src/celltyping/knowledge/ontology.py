"""Cell Ontology (CL) and Uberon access.

* Local OBO copies (via ``obonet``) provide labels, synonyms and the ``is_a``
  hierarchy used to build the classification tree.
* Ubergraph (SPARQL, with OWL reasoning pre-materialised) answers
  "which cell types are located in tissue X", including cells asserted on
  sub-parts of X (``part_of`` is transitive in the redundant graph).
* Everything downloaded or queried is cached under ``data_dir()``.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pandas as pd

from ..utils import data_dir, download, log, norm_label

CL_OBO_URL = "http://purl.obolibrary.org/obo/cl/cl-basic.obo"
UBERON_OBO_URL = "http://purl.obolibrary.org/obo/uberon/basic.obo"
UBERGRAPH_ENDPOINT = "https://ubergraph.apps.renci.org/sparql"
OBO = "http://purl.obolibrary.org/obo/"

CL_ROOT = "CL:0000000"
# Ubergraph (IRIs) uses the RO/BFO ids; the OBO files parsed by obonet key edges by relation *name*.
PART_OF = "BFO:0000050"
LOCATED_IN = "RO:0001025"
HAS_SOMA_LOCATION = "RO:0002100"
OBO_PART_OF = "part_of"

# CL classes that carry no biological signal for marker-based typing; they are
# dropped from the tree and their children re-attached to the grandparent.
ABSTRACT_CL = {
    "CL:0000003",  # native cell
    "CL:0000255",  # eukaryotic cell
    "CL:0002371",  # somatic cell
    "CL:0000548",  # animal cell
    "CL:0002242",  # nucleate cell
    "CL:0000226",  # single nucleate cell
    "CL:0000225",  # anucleate cell
    "CL:0000393",  # electrically responsive cell
    "CL:0000211",  # electrically active cell
    "CL:0000404",  # electrically signaling cell
    "CL:0000151",  # secretory cell
    "CL:0000144",  # cell by function
    "CL:0000010",  # cultured cell
    "CL:0001034",  # cell in vitro
    "CL:0000039",  # germ line cell
    "CL:0000219",  # motile cell
    "CL:0000325",  # stuff accumulating cell
    "CL:0000473",  # defensive cell
    "CL:0000066",  # epithelial cell -- keep? it is informative; do NOT drop
}
ABSTRACT_CL.discard("CL:0000066")

# Uberon terms too generic to be useful as an "ancestor tissue".
GENERIC_UBERON = {
    "UBERON:0000061",  # anatomical structure
    "UBERON:0000062",  # organ
    "UBERON:0000465",  # material anatomical entity
    "UBERON:0001062",  # anatomical entity
    "UBERON:0000468",  # multicellular organism
    "UBERON:0010000",  # multicellular anatomical structure
    "UBERON:0000467",  # anatomical system
    "UBERON:0000064",  # organ part
    "UBERON:0000475",  # organism subdivision
    "UBERON:0000033",  # head
    "UBERON:0013702",  # body proper
    "UBERON:0002050",  # embryonic structure
    "UBERON:0000479",  # tissue
}


# --------------------------------------------------------------------------- OBO loading
def _load_obo(url: str, name: str, force: bool = False) -> nx.MultiDiGraph:
    import obonet

    cache = data_dir() / "ontologies"
    obo_path = download(url, cache / f"{name}.obo", force=force)
    pkl = cache / f"{name}.gpickle"
    if pkl.exists() and pkl.stat().st_mtime >= obo_path.stat().st_mtime and not force:
        with open(pkl, "rb") as fh:
            return pickle.load(fh)
    log.info("parsing %s (first time only)", obo_path.name)
    g = obonet.read_obo(str(obo_path), ignore_obsolete=True)
    with open(pkl, "wb") as fh:
        pickle.dump(g, fh)
    return g


class Ontology:
    """Thin wrapper around an obonet graph (edges point child -> parent, key = relation)."""

    def __init__(self, graph: nx.MultiDiGraph, prefix: str):
        self.g = graph
        self.prefix = prefix
        self._label_index: dict[str, list[str]] | None = None

    # ---- terms
    def __contains__(self, term: str) -> bool:
        return term in self.g

    def label(self, term: str) -> str:
        return self.g.nodes[term].get("name", term) if term in self.g else term

    def synonyms(self, term: str) -> list[str]:
        out = []
        for s in self.g.nodes.get(term, {}).get("synonym", []):
            m = re.match(r'"(.*)"\s+(EXACT|RELATED|NARROW|BROAD)', s)
            if m:
                out.append(m.group(1))
        return out

    def parents(self, term: str, relation: str = "is_a") -> list[str]:
        if term not in self.g:
            return []
        return [v for _, v, k in self.g.out_edges(term, keys=True) if k == relation and v.startswith(self.prefix)]

    def children(self, term: str, relation: str = "is_a") -> list[str]:
        if term not in self.g:
            return []
        return [u for u, _, k in self.g.in_edges(term, keys=True) if k == relation]

    def ancestors(self, term: str, relations: tuple[str, ...] = ("is_a",)) -> set[str]:
        out, stack = set(), [term]
        while stack:
            t = stack.pop()
            for rel in relations:
                for p in self.parents(t, rel):
                    if p not in out:
                        out.add(p)
                        stack.append(p)
        return out

    def descendants(self, term: str, relation: str = "is_a") -> set[str]:
        out, stack = set(), [term]
        while stack:
            t = stack.pop()
            for c in self.children(t, relation):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        return out

    # ---- search
    def _index(self) -> dict[str, list[str]]:
        if self._label_index is None:
            idx: dict[str, list[str]] = {}
            for t, d in self.g.nodes(data=True):
                if not t.startswith(self.prefix):
                    continue
                names = [d.get("name", "")] + self.synonyms(t)
                for nm in names:
                    if nm:
                        idx.setdefault(norm_label(nm), []).append(t)
            self._label_index = idx
        return self._label_index

    def search(self, text: str, exact_only: bool = False) -> list[str]:
        """Return candidate term ids for a free-text label (exact label > synonym > substring)."""
        if re.fullmatch(rf"{self.prefix}\d+", text.strip()):
            return [text.strip()] if text.strip() in self.g else []
        q = norm_label(text)
        idx = self._index()
        hits: list[str] = []
        # exact label first
        for t in idx.get(q, []):
            if norm_label(self.label(t)) == q:
                hits.append(t)
        for t in idx.get(q, []):
            if t not in hits:
                hits.append(t)
        if hits or exact_only:
            return hits
        # substring fallback, shortest labels first
        cands = [(len(k), t) for k, ts in idx.items() if q in k for t in ts]
        return [t for _, t in sorted(set(cands))]


@lru_cache(maxsize=1)
def load_cl(force: bool = False) -> Ontology:
    return Ontology(_load_obo(CL_OBO_URL, "cl-basic", force), "CL:")


@lru_cache(maxsize=1)
def load_uberon(force: bool = False) -> Ontology:
    return Ontology(_load_obo(UBERON_OBO_URL, "uberon-basic", force), "UBERON:")


# --------------------------------------------------------------------------- tissue resolution
@dataclass
class Tissue:
    id: str
    label: str

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.label.lower()).strip("_")


def resolve_tissue(name_or_id: str) -> Tissue:
    """Map a free-text tissue name or ``UBERON:...`` id to a Tissue."""
    ub = load_uberon()
    hits = ub.search(name_or_id)
    if not hits:
        raise ValueError(f"no Uberon term matches '{name_or_id}'")
    if len(hits) > 1 and norm_label(ub.label(hits[0])) != norm_label(name_or_id):
        log.warning("ambiguous tissue '%s'; using %s (%s). Other candidates: %s", name_or_id, hits[0], ub.label(hits[0]),
                    ", ".join(f"{h} ({ub.label(h)})" for h in hits[1:6]))
    return Tissue(hits[0], ub.label(hits[0]))


def tissue_ancestors(uberon_id: str, max_levels: int = 2) -> list[str]:
    """Uberon terms the tissue is ``part_of`` (up to ``max_levels`` hops; ``is_a`` is *not* followed
    because classification parents such as 'ectoderm-derived structure' are not locations)."""
    ub = load_uberon()
    out, frontier = [], [uberon_id]
    for _ in range(max_levels):
        nxt = []
        for t in frontier:
            for p in ub.parents(t, OBO_PART_OF):
                if p not in out and p != uberon_id and p not in GENERIC_UBERON:
                    out.append(p)
                    nxt.append(p)
        frontier = nxt
    return out


def tissue_is_within(uberon_id: str, organ_id: str) -> bool:
    """True if ``uberon_id`` equals, is a subclass of, or is (transitively) part of ``organ_id``.

    ``is_a`` is followed only after at least one ``part_of`` hop or when the terms are direct
    subclasses (e.g. 'cerebral cortex' is_a 'cortex' part_of 'brain'), never through generic classes.
    """
    if uberon_id == organ_id:
        return True
    ub = load_uberon()
    anc = ub.ancestors(uberon_id, ("is_a", OBO_PART_OF))
    if organ_id not in anc:
        return False
    # reject matches that only go through generic classification terms
    if organ_id in GENERIC_UBERON:
        return False
    return True


# --------------------------------------------------------------------------- Ubergraph
def _sparql(query: str) -> list[dict]:
    from SPARQLWrapper import JSON, POST, SPARQLWrapper

    sp = SPARQLWrapper(UBERGRAPH_ENDPOINT)
    sp.setMethod(POST)
    sp.setReturnFormat(JSON)
    sp.setTimeout(120)
    sp.setQuery(query)
    res = sp.query().convert()
    return res["results"]["bindings"]


def _curie(iri: str) -> str:
    return iri.replace(OBO, "").replace("_", ":", 1)


def ubergraph_cells_in_tissue(uberon_id: str, relations: tuple[str, ...] = (PART_OF, LOCATED_IN, HAS_SOMA_LOCATION),
                              force: bool = False) -> pd.DataFrame:
    """CL classes related to ``uberon_id`` (or any of its parts) by one of ``relations``.

    Returns columns ``cl_id, label, relation``. Cached as JSON.
    """
    cache = data_dir() / "ubergraph" / f"{uberon_id.replace(':', '_')}.json"
    if cache.exists() and not force:
        return pd.DataFrame(json.loads(cache.read_text()), columns=["cl_id", "label", "relation"])

    rel_values = " ".join(f"obo:{r.replace(':', '_')}" for r in relations)
    q = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX obo: <{OBO}>
    SELECT DISTINCT ?cell ?label ?rel
    FROM <http://reasoner.renci.org/redundant>
    FROM <http://reasoner.renci.org/ontology>
    WHERE {{
      VALUES ?rel {{ {rel_values} }}
      ?cell rdfs:subClassOf obo:CL_0000000 .
      ?cell ?rel obo:{uberon_id.replace(':', '_')} .
      ?cell rdfs:label ?label .
      FILTER(STRSTARTS(STR(?cell), "{OBO}CL_"))
    }}
    """
    try:
        rows = _sparql(q)
    except Exception as e:  # network failure -> empty result, caller falls back to DB evidence
        log.warning("Ubergraph query failed for %s (%s); continuing without ontology tissue evidence", uberon_id, e)
        return pd.DataFrame(columns=["cl_id", "label", "relation"])
    recs = [[_curie(r["cell"]["value"]), r["label"]["value"], _curie(r["rel"]["value"])] for r in rows]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(recs))
    log.info("Ubergraph: %d cell types related to %s", len(recs), uberon_id)
    return pd.DataFrame(recs, columns=["cl_id", "label", "relation"])


_SPECIES_TAG = re.compile(r"\s*\((Mmus|Hsap|Homo sapiens|Mus musculus|Primate)\)\s*$")


def expected_cell_types(tissue: Tissue, species: str = "human", ancestor_levels: int = 1) -> pd.DataFrame:
    """Ontology-derived expected cell types for a tissue and (optionally) its parent tissues/systems.

    Returns columns ``cl_id, label, relation, tissue_id, tissue_label``.
    Species-tagged provisional terms of the other species are removed.
    """
    frames = []
    ids = [tissue.id] + (tissue_ancestors(tissue.id, ancestor_levels) if ancestor_levels > 0 else [])
    ub = load_uberon()
    for tid in ids:
        df = ubergraph_cells_in_tissue(tid)
        df = df.assign(tissue_id=tid, tissue_label=ub.label(tid))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["cl_id", "label", "relation", "tissue_id", "tissue_label"])
    if out.empty:
        return out
    out = out[~out["label"].map(lambda l: is_other_species_term(l, species))]
    out["label"] = out["label"].str.replace(_SPECIES_TAG, "", regex=True)
    return out.drop_duplicates("cl_id").reset_index(drop=True)


def is_other_species_term(label: str, species: str = "human") -> bool:
    """True for provisional CL terms tagged with a species other than ``species`` (e.g. '... (Mmus)')."""
    m = _SPECIES_TAG.search(label or "")
    if not m:
        return False
    tag = m.group(1)
    human = species.lower().startswith("h")
    return (tag in ("Mmus", "Mus musculus")) if human else (tag in ("Hsap", "Homo sapiens"))


def clean_label(label: str) -> str:
    return _SPECIES_TAG.sub("", label or "").strip()


# --------------------------------------------------------------------------- CL -> tree
def induced_is_a_dag(cl: Ontology, seeds: set[str]) -> nx.DiGraph:
    """DAG (child -> parent) over ``seeds`` and all their is_a ancestors up to CL root."""
    g = nx.DiGraph()
    todo = [s for s in seeds if s in cl]
    seen = set()
    while todo:
        t = todo.pop()
        if t in seen:
            continue
        seen.add(t)
        g.add_node(t)
        for p in cl.parents(t, "is_a"):
            g.add_edge(t, p)
            todo.append(p)
    g.add_node(CL_ROOT)
    return g


def dag_to_tree_parents(dag: nx.DiGraph, prefer: set[str], forced: dict[str, str] | None = None) -> dict[str, str | None]:
    """Choose a single parent per node.

    Priority: forced override > non-abstract parent > parent in ``prefer`` (tissue-evidenced) >
    parent with the longest path from the root (most specific).
    """
    forced = forced or {}
    # depth = longest path from root (edges child->parent so we walk reversed)
    rev = dag.reverse(copy=True)
    depth = {CL_ROOT: 0}
    for n in nx.topological_sort(rev):
        for c in rev.successors(n):
            depth[c] = max(depth.get(c, 0), depth.get(n, 0) + 1)
    parents: dict[str, str | None] = {CL_ROOT: None}
    for n in dag.nodes:
        if n == CL_ROOT:
            continue
        cands = list(dag.successors(n))
        if n in forced and forced[n] in dag.nodes:
            parents[n] = forced[n]
            continue
        if not cands:
            parents[n] = CL_ROOT
            continue
        cands.sort(key=lambda p: (p not in ABSTRACT_CL, p in prefer, depth.get(p, 0), p), reverse=True)
        parents[n] = cands[0]
    return parents
