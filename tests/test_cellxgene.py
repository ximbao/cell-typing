"""Offline tests for the CZ CELLxGENE knowledge source (no network)."""

import pandas as pd

from celltyping.knowledge.cellxgene import aggregate_census_obs, census_seeds, parse_cellguide_markers


def _obs():
    return pd.DataFrame({
        "cell_type_ontology_term_id": ["CL:0000057"] * 4 + ["CL:0000235"] * 2 + ["CL:0000001"] * 3 + ["CL:0000501"],
        "cell_type": ["fibroblast"] * 4 + ["macrophage"] * 2 + ["primary cultured cell"] * 3 + ["granulosa cell"],
        "dataset_id": ["d1", "d1", "d2", "d3", "d1", "d1", "d1", "d2", "d3", "d1"],
        "disease": ["normal", "normal", "cancer", "normal", "normal", "cancer", "normal", "normal", "normal", "normal"],
    })


def test_aggregate_census_obs():
    df = aggregate_census_obs(_obs())
    fb = df.set_index("cl_id").loc["CL:0000057"]
    assert fb["n_cells"] == 4 and fb["n_datasets"] == 3 and fb["n_diseases"] == 2 and fb["n_cells_normal"] == 3
    assert fb["n_datasets_normal"] == 2  # d2 is cancer-only
    assert list(df["cl_id"])[0] == "CL:0000057"  # sorted by n_cells desc


def test_census_seeds_thresholds_and_ignore():
    df = aggregate_census_obs(_obs())
    seeds = census_seeds(df, min_cells=2, min_datasets=2, min_frac=0.0)
    ids = set(seeds["cl_id"])
    assert "CL:0000057" in ids
    assert "CL:0000235" not in ids  # only one dataset
    assert "CL:0000501" not in ids  # one cell
    assert "CL:0000001" not in ids  # primary cultured cell is ignored even though it passes thresholds


def _entries():
    def e(sym, score, spec, pc, tissue=None, org="Homo sapiens"):
        gd = {"organism_ontology_term_label": org}
        if tissue:
            gd["tissue_ontology_term_label"] = tissue
        return {"symbol": sym, "marker_score": score, "specificity": spec, "pc": pc, "me": 1.0, "groupby_dims": gd}

    return [
        e("AMH", 2.0, 0.95, 0.6, "ovary"), e("FOXL2", 1.5, 0.9, 0.5, "ovary"), e("LOW", 1.0, 0.3, 0.5, "ovary"),
        e("RARE", 1.8, 0.9, 0.01, "ovary"),
        e("INHA", 1.0, 0.9, 0.5), e("C7_ENSG00000112936", 5.0, 0.9, 0.5), e("Amh", 2.0, 0.9, 0.5, org="Mus musculus"),
    ]


def test_parse_cellguide_prefers_tissue_entry_and_filters():
    df = parse_cellguide_markers(_entries(), "CL:0000501", "human", "ovary", "UBERON:0000992")
    assert set(df["gene"]) == {"AMH", "FOXL2"}  # LOW fails specificity, RARE fails pc, INHA is organism-wide only
    assert (df["tissue_label"] == "ovary").all() and (df["source"] == "CellGuide").all()
    w = df.set_index("gene")["weight"]
    assert w["AMH"] == 2.0 and w["FOXL2"] == 1.5  # absolute marker_score


def test_parse_cellguide_falls_back_to_organism_wide():
    df = parse_cellguide_markers(_entries(), "CL:0000501", "human", "lung", None)
    assert list(df["gene"]) == ["C7", "INHA"] and df["tissue_label"].isna().all()  # Ensembl suffix stripped
    assert df.set_index("gene")["weight"]["C7"] == 2.0  # clipped at max_weight
    assert parse_cellguide_markers(_entries(), "CL:0000501", "human", None).shape[0] == 2
    assert parse_cellguide_markers([], "CL:0000501").empty


def test_census_seeds_min_frac():
    df = pd.DataFrame({"cl_id": ["CL:0000057", "CL:0000232"], "label": ["fibroblast", "erythrocyte"],
                       "n_cells": [100_000, 500], "n_datasets": [5, 3], "n_diseases": [1, 1], "n_cells_normal": [100_000, 500]})
    assert list(census_seeds(df, min_cells=200, min_datasets=2, min_frac=0.01)["cl_id"]) == ["CL:0000057"]
    assert len(census_seeds(df, min_cells=200, min_datasets=2, min_frac=0.0)) == 2
