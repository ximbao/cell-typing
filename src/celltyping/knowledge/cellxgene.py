"""CZ CELLxGENE as a knowledge source.

Two complementary pieces:

* **Census** (``cellxgene_census`` API): which Cell Ontology types have actually been observed in a
  tissue, across how many datasets, in how many cells. Expert-annotated, ontology-normalised metadata;
  used as tissue evidence for seeding the tree. Requires the optional ``cellxgene-census`` package;
  results are cached per (version, tissue_general) so the API is hit once.
* **CellGuide** computational marker genes (static JSON, no extra dependency): per cell type, per
  organism and per ``tissue_general``, up to 100 genes with ``marker_score`` / ``specificity`` /
  fraction expressing, computed from the Census. Used as a marker source (``source="CellGuide"``).

CellGuide's *canonical* markers are HuBMAP ASCT+B rows, which ``markers.py`` already loads.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..utils import data_dir, download, log, norm_gene
from .markers import COLUMNS, _empty
from .ontology import ABSTRACT_CL, Tissue, is_other_species_term, load_uberon, tissue_ancestors, tissue_is_within

CELLGUIDE_BASE = "https://cellguide.cellxgene.cziscience.com"
DEFAULT_CENSUS_VERSION = "2025-01-30"  # stable LTS; pin for reproducibility
ORGANISM = {"human": "Homo sapiens", "mouse": "Mus musculus"}
CENSUS_ORGANISM = {"human": "homo_sapiens", "mouse": "mus_musculus"}

# Frequent Census annotations that are not typing targets: the root, culture / in-vitro / embryonic
# states and 'hematopoietic cell' (too coarse; it enters the tree as an ancestor when needed anyway).
CENSUS_IGNORE = ABSTRACT_CL | {"CL:0000000", "CL:0000001", "CL:0000578", "CL:0000007", "CL:0002321", "CL:0002322", "CL:0000988"}


# --------------------------------------------------------------------------- tissue_general
@lru_cache(maxsize=1)
def cellguide_snapshot() -> str:
    p = data_dir() / "cellxgene" / "cellguide_snapshot.txt"
    if p.exists() and p.stat().st_size:
        return p.read_text().strip()
    import requests

    r = requests.get(f"{CELLGUIDE_BASE}/latest_snapshot_identifier", timeout=60)
    r.raise_for_status()
    snap = r.text.strip()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(snap)
    return snap


@lru_cache(maxsize=1)
def census_tissues() -> pd.DataFrame:
    """The ~66 UBERON terms that define Census ``tissue_general`` (from CellGuide's tissue metadata)."""
    p = download(f"{CELLGUIDE_BASE}/{cellguide_snapshot()}/tissue_metadata.json", data_dir() / "cellxgene" / "tissue_metadata.json")
    d = json.loads(Path(p).read_text())
    return pd.DataFrame([{"id": k, "label": v.get("name", k)} for k, v in d.items()])


def census_tissue_general(tissue: Tissue) -> tuple[str, str] | None:
    """Map a tissue to the Census ``tissue_general`` category it falls in (exact, then nearest ancestor)."""
    tg = census_tissues()
    ids = set(tg["id"])
    label = dict(zip(tg["id"], tg["label"]))
    if tissue.id in ids:
        return tissue.id, label[tissue.id]
    for a in tissue_ancestors(tissue.id, max_levels=6):
        if a in ids:
            return a, label[a]
    # fall back to is_a/part_of containment (e.g. 'lamina propria' within 'intestine')
    for tid in tg["id"]:
        if tissue_is_within(tissue.id, tid):
            return tid, label[tid]
    return None


# --------------------------------------------------------------------------- Census: cell types in a tissue
def census_cell_types(tissue_general: str, species: str = "human", census_version: str = DEFAULT_CENSUS_VERSION,
                      disease: str = "any", force: bool = False) -> pd.DataFrame:
    """Cell types observed in a Census ``tissue_general`` category.

    Returns ``cl_id, label, n_cells, n_datasets, n_diseases, n_cells_normal, n_datasets_normal`` (primary cells only).
    Cached under ``data/cellxgene/census-<version>/<organism>-<tissue_general>.csv``. ``disease='normal'`` restricts
    ``n_cells`` / ``n_datasets`` to healthy samples (Census ``disease == 'normal'``).
    Returns an empty frame (with a warning) if ``cellxgene_census`` is not installed or the query fails.
    """
    org = CENSUS_ORGANISM.get(species.lower())
    if org is None:
        log.warning("Census: no organism for species %r", species)
        return pd.DataFrame(columns=["cl_id", "label", "n_cells", "n_datasets", "n_diseases", "n_cells_normal"])
    slug = tissue_general.lower().replace(" ", "_").replace("/", "_")
    cache = data_dir() / "cellxgene" / f"census-{census_version}" / f"{org}-{slug}.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache)
    else:
        try:
            import cellxgene_census
        except ImportError:
            log.warning("cellxgene_census not installed (pip install 'celltyping[census]'); Census tissue evidence skipped")
            return pd.DataFrame(columns=["cl_id", "label", "n_cells", "n_datasets", "n_diseases", "n_cells_normal"])
        log.info("Census %s: fetching cell types for tissue_general=%r (%s) ...", census_version, tissue_general, org)
        try:
            with cellxgene_census.open_soma(census_version=census_version) as census:
                obs = cellxgene_census.get_obs(
                    census, org,
                    value_filter=f"tissue_general == '{tissue_general}' and is_primary_data == True",
                    column_names=["cell_type_ontology_term_id", "cell_type", "dataset_id", "disease"],
                )
        except Exception as e:  # network / version problems should not break the build
            log.warning("Census query failed (%s); Census tissue evidence skipped", e)
            return pd.DataFrame(columns=["cl_id", "label", "n_cells", "n_datasets", "n_diseases", "n_cells_normal"])
        df = aggregate_census_obs(obs)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        log.info("Census: %d cells, %d cell types in %r (cached %s)", int(df["n_cells"].sum()), len(df), tissue_general, cache)
    if disease == "normal":
        df = df[df["n_cells_normal"] > 0].copy()
        df["n_cells"] = df["n_cells_normal"]
        if "n_datasets_normal" in df:
            df["n_datasets"] = df["n_datasets_normal"]
    return df


def aggregate_census_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a Census ``obs`` slice (cell_type_ontology_term_id, cell_type, dataset_id, disease) per cell type."""
    obs = obs.copy()
    obs["is_normal"] = obs["disease"].astype(str).eq("normal")
    g = obs.groupby(["cell_type_ontology_term_id", "cell_type"], observed=True)
    out = g.agg(n_cells=("dataset_id", "size"), n_datasets=("dataset_id", "nunique"), n_diseases=("disease", "nunique"),
                n_cells_normal=("is_normal", "sum")).reset_index()
    nd_normal = obs[obs["is_normal"]].groupby("cell_type_ontology_term_id", observed=True)["dataset_id"].nunique()
    out["n_datasets_normal"] = out["cell_type_ontology_term_id"].map(nd_normal).fillna(0).astype(int)
    out = out.rename(columns={"cell_type_ontology_term_id": "cl_id", "cell_type": "label"})
    out["cl_id"] = out["cl_id"].astype(str).str.replace("_", ":")
    return out.sort_values("n_cells", ascending=False).reset_index(drop=True)


def census_seeds(df: pd.DataFrame, species: str = "human", min_cells: int = 200, min_datasets: int = 2,
                 min_frac: float = 0.001) -> pd.DataFrame:
    """Rows of :func:`census_cell_types` strong enough to seed the tree.

    A type needs ``>= min_cells`` cells, ``>= min_datasets`` datasets and ``>= min_frac`` of all cells in the tissue
    (so that in a 20M-cell tissue like brain, a few thousand contaminating erythrocytes or neutrophils do not count);
    abstract / culture / other-species terms are removed.
    """
    if df.empty:
        return df
    total = float(df["n_cells"].sum())
    keep = (df["n_cells"] >= min_cells) & (df["n_datasets"] >= min_datasets) & df["cl_id"].str.startswith("CL:")
    keep &= df["n_cells"] / max(total, 1.0) >= min_frac
    keep &= ~df["cl_id"].isin(CENSUS_IGNORE)
    keep &= ~df["label"].map(lambda l: is_other_species_term(str(l), species))
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- CellGuide computational markers
def _cellguide_file(cl_id: str, force: bool = False) -> Path | None:
    snap = cellguide_snapshot()
    name = cl_id.replace(":", "_")
    dest = data_dir() / "cellxgene" / "cellguide" / snap / f"{name}.json"
    try:
        return download(f"{CELLGUIDE_BASE}/{snap}/computational_marker_genes/{name}.json", dest, force=force)
    except Exception as e:  # cell type without a CellGuide page
        log.debug("CellGuide: no marker file for %s (%s)", cl_id, e)
        return None


def parse_cellguide_markers(entries: list[dict], cl_id: str, species: str = "human", tissue_general: str | None = None,
                            tissue_id: str | None = None, min_specificity: float = 0.7, min_pc: float = 0.1,
                            top_n: int = 50, max_weight: float = 2.0) -> pd.DataFrame:
    """Turn one CellGuide ``computational_marker_genes`` file into marker rows (``markers.COLUMNS``).

    Prefers the (organism, tissue_general) entry; falls back to the organism-wide entry. ``weight`` is the
    absolute ``marker_score`` (an effect size; CellGuide exports genes with score >= 0.5, strong markers reach 2-3),
    clipped at ``max_weight``, so weakly marked types are not inflated relative to well-marked ones.
    """
    organism = ORGANISM.get(species.lower(), species)
    org_rows = [e for e in entries if e.get("groupby_dims", {}).get("organism_ontology_term_label") == organism]
    tissue_rows = [e for e in org_rows if e["groupby_dims"].get("tissue_ontology_term_label") == tissue_general] if tissue_general else []
    all_rows = [e for e in org_rows if "tissue_ontology_term_label" not in e["groupby_dims"]]
    chosen, scope = (tissue_rows, tissue_general) if tissue_rows else (all_rows, None)
    chosen = [e for e in chosen if float(e.get("specificity", 0)) >= min_specificity and float(e.get("pc", 0)) >= min_pc
              and e.get("symbol")]
    if not chosen:
        return _empty()
    chosen = sorted(chosen, key=lambda e: -float(e["marker_score"]))[:top_n]
    rows = []
    for e in chosen:
        w = min(float(e["marker_score"]), max_weight)
        rows.append({"cl_id": cl_id, "cell_label": None, "gene": _symbol(e["symbol"]), "source": "CellGuide",
                     "tissue_id": tissue_id if scope else None, "tissue_label": scope, "species": species, "weight": round(w, 4)})
    return pd.DataFrame(rows, columns=COLUMNS)


_ENSEMBL_SUFFIX = re.compile(r"_ENS[A-Z]*G\d+$")


def _symbol(sym: str) -> str:
    """CellGuide disambiguates duplicated symbols as ``C7_ENSG00000112936``; keep the symbol only."""
    return norm_gene(_ENSEMBL_SUFFIX.sub("", str(sym)))


def cellguide_markers(cl_ids: list[str], species: str = "human", tissue_general: tuple[str, str] | None = None,
                      force: bool = False, **parse_kw) -> pd.DataFrame:
    """CellGuide computational markers for the given cell types (files are downloaded once and cached)."""
    tid, tlabel = tissue_general if tissue_general else (None, None)
    frames, n_tissue = [], 0
    for cl_id in cl_ids:
        if not str(cl_id).startswith("CL:"):
            continue
        p = _cellguide_file(cl_id, force=force)
        if p is None:
            continue
        try:
            entries = json.loads(Path(p).read_text())
        except Exception:
            continue
        df = parse_cellguide_markers(entries, cl_id, species, tlabel, tid, **parse_kw)
        if not df.empty:
            n_tissue += int(df["tissue_label"].notna().any())
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else _empty()
    log.info("CellGuide: markers for %d / %d cell types (%d with %s-specific entries)", len(frames), len(cl_ids), n_tissue, tlabel or "tissue")
    return out
