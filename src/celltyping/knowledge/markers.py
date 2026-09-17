"""Marker-gene knowledge sources, harmonised to Cell Ontology ids.

Each loader returns a long ``pandas.DataFrame`` with columns::

    cl_id, cell_label, gene, source, tissue_id, tissue_label, species, weight

* **CellMarker 2.0** (manually curated; rows carry CL and UBERON ids).
* **PanglaoDB** (curated; organ + cell-type labels, mapped to CL by label/synonym).
* **HuBMAP ASCT+B** (expert-authored organ tables; CL ids + HGNC gene biomarkers).

Weights encode per-source confidence and are summed across sources downstream.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import data_dir, download, log, norm_gene, norm_label
from .ontology import Tissue, load_cl, resolve_tissue, tissue_is_within

COLUMNS = ["cl_id", "cell_label", "gene", "source", "tissue_id", "tissue_label", "species", "weight"]

CELLMARKER_URLS = {
    "human": "https://bio-bigdata.hrbmu.edu.cn/CellMarker2.0/CellMarker_download_files/file/Cell_marker_Human.xlsx",
    "mouse": "https://bio-bigdata.hrbmu.edu.cn/CellMarker2.0/CellMarker_download_files/file/Cell_marker_Mouse.xlsx",
}
PANGLAO_URL = "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"
ASCTB_CDN = "https://cdn.humanatlas.io/digital-objects/asct-b/{organ}/latest/assets/asct-b-{prefix}{organ}.csv"

# HuBMAP ASCT+B organ tables -> Uberon id of the organ they describe.
ASCTB_ORGANS: dict[str, str] = {
    "allen-brain": "UBERON:0000955",
    "spinal-cord": "UBERON:0002240",
    "peripheral-nervous-system": "UBERON:0000010",
    "eye": "UBERON:0000970",
    "heart": "UBERON:0000948",
    "lung": "UBERON:0002048",
    "trachea": "UBERON:0003126",
    "main-bronchus": "UBERON:0002182",
    "kidney": "UBERON:0002113",
    "ureter": "UBERON:0000056",
    "urinary-bladder": "UBERON:0001255",
    "liver": "UBERON:0002107",
    "pancreas": "UBERON:0001264",
    "small-intestine": "UBERON:0002108",
    "large-intestine": "UBERON:0000059",
    "mouth": "UBERON:0000165",
    "palatine-tonsil": "UBERON:0002373",
    "spleen": "UBERON:0002106",
    "thymus": "UBERON:0002370",
    "lymph-node": "UBERON:0000029",
    "bone-marrow": "UBERON:0002371",
    "skin": "UBERON:0002097",
    "prostate": "UBERON:0002367",
    "uterus": "UBERON:0000995",
    "ovary": "UBERON:0000992",
    "fallopian-tube": "UBERON:0003889",
    "placenta": "UBERON:0001987",
    "knee": "UBERON:0001465",
    "skeleton": "UBERON:0004288",
    "muscular-system": "UBERON:0000383",
    "blood-vasculature": "UBERON:0004537",
    "lymph-vasculature": "UBERON:0001473",
}
# Organs whose cell types are expected in essentially every solid tissue section.
ASCTB_GENERIC_ORGANS = ("blood-vasculature",)

# PanglaoDB / free-text cell-type labels that do not resolve to CL by label search.
LABEL_TO_CL: dict[str, str] = {
    "microglia": "CL:0000129",
    "oligodendrocyte progenitor cell": "CL:0002453",
    "opc": "CL:0002453",
    "neural stem precursor cell": "CL:0000047",
    "glutaminergic neuron": "CL:0000679",
    "glutamatergic neuron": "CL:0000679",
    "gabaergic neuron": "CL:0000617",
    "interneuron": "CL:0000099",
    "endothelial cell": "CL:0000115",
    "pericyte": "CL:0000669",
    "fibroblast": "CL:0000057",
    "smooth muscle cell": "CL:0000192",
    "vascular smooth muscle cell": "CL:0000359",
    "macrophage": "CL:0000235",
    "monocyte": "CL:0000576",
    "t cell": "CL:0000084",
    "b cell": "CL:0000236",
    "nk cell": "CL:0000623",
    "natural killer cell": "CL:0000623",
    "dendritic cell": "CL:0000451",
    "plasma cell": "CL:0000786",
    "mast cell": "CL:0000097",
    "neutrophil": "CL:0000775",
    "erythroid like and erythroid precursor cell": "CL:0000764",
    "red blood cell": "CL:0000232",
    "ependymal cell": "CL:0000065",
    "choroid plexus cell": "CL:0000706",
    "purkinje neuron": "CL:0000121",
    "bergmann glia": "CL:0000644",
    "schwann cell": "CL:0002573",
    "radial glia cell": "CL:0000681",
    "neuroblast": "CL:0000031",
    "neuroendocrine cell": "CL:0000165",
    "epithelial cell": "CL:0000066",
    "adipocyte": "CL:0000136",
    "myofibroblast": "CL:0000186",
    "lymphatic endothelial cell": "CL:0002138",
}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _cl_curie(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().replace("_", ":")
    return s if re.fullmatch(r"CL:\d{7}", s) else None


def map_label_to_cl(label: str) -> str | None:
    """Best-effort mapping of a free-text cell-type label to a CL id."""
    key = norm_label(label)
    if key in LABEL_TO_CL:
        return LABEL_TO_CL[key]
    cl = load_cl()
    hits = cl.search(label, exact_only=True)
    if hits:
        return hits[0]
    # try singular "X cell" form
    if not key.endswith("cell"):
        hits = cl.search(key + " cell", exact_only=True)
        if hits:
            return hits[0]
    return None


# --------------------------------------------------------------------------- CellMarker 2.0
def load_cellmarker(species: str = "human", force: bool = False) -> pd.DataFrame:
    """Full CellMarker 2.0 table for a species (all tissues), harmonised."""
    sp = species.lower()
    url = CELLMARKER_URLS["human" if sp.startswith("h") else "mouse"]
    cache = data_dir() / "markers" / Path(url).name
    parquet = cache.with_suffix(".parquet")
    if parquet.exists() and not force:
        return pd.read_parquet(parquet)
    try:
        download(url, cache, force=force)
    except Exception as e:
        log.warning("CellMarker download failed (%s); source skipped", e)
        return _empty()
    raw = pd.read_excel(cache)
    raw.columns = [c.strip() for c in raw.columns]
    ub_col = next((c for c in raw.columns if c.lower().startswith("uberon")), None)
    df = pd.DataFrame({
        "cl_id": raw["cellontology_id"].map(_cl_curie),
        "cell_label": raw["cell_name"].astype(str),
        "gene": raw["Symbol"].where(raw["Symbol"].notna(), raw["marker"]).map(norm_gene),
        "tissue_id": raw[ub_col].astype(str).str.replace("_", ":", regex=False) if ub_col else None,
        "tissue_label": raw["tissue_class"].astype(str) if "tissue_class" in raw else raw.get("tissue_type", "").astype(str),
        "cell_type": raw.get("cell_type", "Normal cell"),
        "pmid": raw.get("PMID", ""),
    })
    df = df[df["cl_id"].notna() & df["gene"].notna() & (df["gene"] != "NAN")]
    df = df[df["cell_type"].astype(str).str.lower().str.contains("normal")]  # drop cancer-cell rows
    # weight: 1 + up to 0.5 for literature support breadth
    grp = df.groupby(["cl_id", "gene", "tissue_id"], dropna=False)
    agg = grp.agg(cell_label=("cell_label", "first"), tissue_label=("tissue_label", "first"), n_ref=("pmid", "nunique")).reset_index()
    agg["weight"] = 1.0 + 0.5 * np.clip(np.log1p(agg["n_ref"]) / np.log1p(5), 0, 1)
    agg["source"] = "CellMarker2"
    agg["species"] = "human" if sp.startswith("h") else "mouse"
    out = agg[COLUMNS]
    out.to_parquet(parquet)
    return out


# --------------------------------------------------------------------------- PanglaoDB
def load_panglao(species: str = "human", force: bool = False) -> pd.DataFrame:
    sp = species.lower()
    cache = data_dir() / "markers" / "PanglaoDB_markers_27_Mar_2020.tsv.gz"
    parquet = cache.with_name(f"panglao_{sp[:2]}.parquet")
    if parquet.exists() and not force:
        return pd.read_parquet(parquet)
    try:
        download(PANGLAO_URL, cache, force=force)
    except Exception as e:
        log.warning("PanglaoDB download failed (%s); source skipped", e)
        return _empty()
    raw = pd.read_csv(cache, sep="\t", compression="gzip")
    raw.columns = [c.strip() for c in raw.columns]
    tag = "Hs" if sp.startswith("h") else "Mm"
    raw = raw[raw["species"].astype(str).str.contains(tag)]
    ub = {}
    tissue_ids, tissue_labels = [], []
    for organ in raw["organ"].fillna("").astype(str):
        if organ not in ub:
            try:
                t = resolve_tissue(organ) if organ else None
            except ValueError:
                t = None
            ub[organ] = t
        tissue_ids.append(ub[organ].id if ub[organ] else None)
        tissue_labels.append(organ)
    cl_map = {lab: map_label_to_cl(lab) for lab in raw["cell type"].unique()}
    unmapped = sorted(k for k, v in cl_map.items() if v is None)
    if unmapped:
        log.info("PanglaoDB: %d/%d cell-type labels not mapped to CL (e.g. %s)", len(unmapped), len(cl_map), ", ".join(unmapped[:8]))
    df = pd.DataFrame({
        "cl_id": raw["cell type"].map(cl_map),
        "cell_label": raw["cell type"].astype(str),
        "gene": raw["official gene symbol"].map(norm_gene),
        "tissue_id": tissue_ids,
        "tissue_label": tissue_labels,
        "canonical": raw.get("canonical marker", 0).fillna(0).astype(float),
        "ubiq": raw.get("ubiquitousness index", 0).fillna(0).astype(float),
    })
    df = df[df["cl_id"].notna()]
    # weight: 1, +0.5 if canonical, minus penalty for ubiquitous genes
    df["weight"] = 1.0 + 0.5 * (df["canonical"] > 0) - 0.5 * np.clip(df["ubiq"], 0, 1)
    df["source"] = "PanglaoDB"
    df["species"] = "human" if sp.startswith("h") else "mouse"
    out = df[COLUMNS].drop_duplicates(["cl_id", "gene", "tissue_id"])
    out.to_parquet(parquet)
    return out


# --------------------------------------------------------------------------- HuBMAP ASCT+B
def _asctb_url(organ: str) -> str:
    prefix = "" if organ == "allen-brain" else "vh-"
    return ASCTB_CDN.format(organ=organ, prefix=prefix)


def _parse_asctb_csv(text: str, organ: str, organ_uberon: str) -> pd.DataFrame:
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("AS/1")), None)
    if start is None:
        return _empty()
    rdr = csv.reader(io.StringIO("\n".join(lines[start:])))
    header = next(rdr)
    col = {h: i for i, h in enumerate(header)}
    ct_ids = sorted((h for h in header if re.fullmatch(r"CT/\d+/ID", h)), key=lambda h: int(h.split("/")[1]))
    gene_cols = sorted((h for h in header if re.fullmatch(r"BGene/\d+", h)), key=lambda h: int(h.split("/")[1]))
    as_ids = sorted((h for h in header if re.fullmatch(r"AS/\d+/ID", h)), key=lambda h: int(h.split("/")[1]))
    recs = []
    for row in rdr:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        # most specific cell type in the row
        ct = None
        for h in reversed(ct_ids):
            cid = _cl_curie(row[col[h]])
            if cid:
                lab = row[col[h.replace("/ID", "/LABEL")]] or row[col[h.replace("/ID", "")]]
                ct = (cid, lab)
                break
        if ct is None:
            continue
        # most specific anatomical structure with an UBERON id
        as_id = organ_uberon
        for h in reversed(as_ids):
            v = row[col[h]].strip()
            if re.fullmatch(r"UBERON:\d+", v):
                as_id = v
                break
        genes = {norm_gene(row[col[g]]) for g in gene_cols if row[col[g]].strip()}
        for g in genes:
            recs.append([ct[0], ct[1], g, "ASCT+B", as_id, organ, "human", 1.5])
    return pd.DataFrame(recs, columns=COLUMNS).drop_duplicates(["cl_id", "gene", "tissue_id"])


def load_asctb(organs: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    """HuBMAP ASCT+B tables (human) for the given organ slugs (default: all known)."""
    organs = organs or list(ASCTB_ORGANS)
    frames = []
    for organ in organs:
        if organ not in ASCTB_ORGANS:
            log.warning("unknown ASCT+B organ '%s'", organ)
            continue
        cache = data_dir() / "markers" / "asctb" / f"{organ}.csv"
        try:
            download(_asctb_url(organ), cache, force=force)
        except Exception as e:
            log.warning("ASCT+B '%s' download failed (%s); skipped", organ, e)
            continue
        frames.append(_parse_asctb_csv(cache.read_text(errors="replace"), organ, ASCTB_ORGANS[organ]))
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def asctb_organs_for_tissue(tissue: Tissue, include_generic: bool = True) -> list[str]:
    """ASCT+B organ tables relevant to a tissue: the organ containing the tissue, organs contained in it, plus generic ones."""
    out = []
    for organ, uid in ASCTB_ORGANS.items():
        if tissue_is_within(tissue.id, uid) or tissue_is_within(uid, tissue.id):
            out.append(organ)
    if include_generic:
        out += [o for o in ASCTB_GENERIC_ORGANS if o not in out]
    return out


# --------------------------------------------------------------------------- user overrides
def load_custom_markers(path: str | Path, species: str = "human") -> pd.DataFrame:
    """User-supplied markers from CSV/TSV (columns: cl_id or cell_label, gene[, weight]) or YAML ``{cl_id: [genes]}``."""
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        d = yaml.safe_load(path.read_text()) or {}
        recs = []
        for key, genes in d.items():
            cid = _cl_curie(key) or map_label_to_cl(key)
            if cid is None:
                log.warning("custom markers: cannot map '%s' to CL; skipped", key)
                continue
            for g in genes:
                recs.append([cid, key, norm_gene(g), "custom", None, None, species, 2.0])
        return pd.DataFrame(recs, columns=COLUMNS)
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    if "cl_id" not in df:
        df["cl_id"] = df["cell_label"].map(map_label_to_cl)
    df = df[df["cl_id"].notna()]
    return pd.DataFrame({
        "cl_id": df["cl_id"].map(_cl_curie),
        "cell_label": df.get("cell_label", df["cl_id"]),
        "gene": df["gene"].map(norm_gene),
        "source": "custom",
        "tissue_id": None,
        "tissue_label": None,
        "species": species,
        "weight": df.get("weight", 2.0),
    })[COLUMNS]


# --------------------------------------------------------------------------- aggregate
def markers_for_tissue(tissue: Tissue, species: str = "human", sources: tuple[str, ...] = ("CellMarker2", "PanglaoDB", "ASCT+B"),
                       tissue_ids: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    """Marker rows from all requested sources whose tissue annotation falls within ``tissue`` (or ``tissue_ids``).

    Rows without tissue annotation are kept (they are attached only to cell types that are
    otherwise expected in the tissue). A column ``tissue_match`` marks rows that matched.
    """
    scope = set(tissue_ids or []) | {tissue.id}
    frames = []
    if "CellMarker2" in sources:
        frames.append(load_cellmarker(species, force))
    if "PanglaoDB" in sources:
        frames.append(load_panglao(species, force))
    if "ASCT+B" in sources:
        organs = asctb_organs_for_tissue(tissue)
        log.info("ASCT+B organ tables selected for %s: %s", tissue.label, ", ".join(organs) or "none")
        frames.append(load_asctb(organs, force))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else _empty()
    if df.empty:
        return df.assign(tissue_match=False)
    cache: dict[str, bool] = {}

    def _match(tid) -> bool:
        if tid is None or (isinstance(tid, float) and np.isnan(tid)) or not str(tid).startswith("UBERON:"):
            return False
        if tid not in cache:
            cache[tid] = any(tissue_is_within(tid, s) for s in scope) or tissue_is_within(tissue.id, tid)
        return cache[tid]

    df["tissue_match"] = df["tissue_id"].map(_match)
    return df
