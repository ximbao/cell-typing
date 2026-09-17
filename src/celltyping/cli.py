"""Command-line interface: ``celltyping --help``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(help="Ontology-driven hierarchical cell typing for imaging-based spatial transcriptomics.", no_args_is_help=True)


@app.command("search-tissue")
def search_tissue(query: str, n: int = 10):
    """List Uberon terms matching a tissue name (to pick the right one for --tissue)."""
    from .knowledge.ontology import load_uberon

    ub = load_uberon()
    hits = ub.search(query)[:n]
    if not hits:
        rprint(f"[red]no match for '{query}'")
        raise typer.Exit(1)
    t = Table("UBERON id", "label", "synonyms")
    for h in hits:
        t.add_row(h, ub.label(h), "; ".join(ub.synonyms(h)[:4]))
    rprint(t)


@app.command("expected-cell-types")
def expected(tissue: str, species: str = "human", ancestor_levels: int = 1,
             census: bool = typer.Option(True, help="Also list cell types observed in the tissue in the CZ CELLxGENE Census"),
             census_min_cells: int = 200, census_min_datasets: int = 2, census_disease: str = "normal"):
    """Show cell types the ontology (and the CELLxGENE Census) place in a tissue, before any marker filtering."""
    from .knowledge.cellxgene import census_cell_types, census_seeds, census_tissue_general
    from .knowledge.ontology import expected_cell_types, resolve_tissue

    t = resolve_tissue(tissue)
    df = expected_cell_types(t, species, ancestor_levels)
    cx = {}
    if census:
        tg = census_tissue_general(t)
        if tg is None:
            rprint("[yellow]no Census tissue_general category contains this tissue[/yellow]")
        else:
            cxdf = census_seeds(census_cell_types(tg[1], species, disease=census_disease), species, census_min_cells, census_min_datasets)
            cx = {r.cl_id: (r.label, int(r.n_cells), int(r.n_datasets)) for r in cxdf.itertuples()}
            extra = [{"cl_id": k, "label": v[0], "relation": "-", "tissue_id": tg[0], "tissue_label": f"Census:{tg[1]}"}
                     for k, v in cx.items() if k not in set(df["cl_id"])]
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True) if extra else df
    rprint(f"[bold]{t.label} ({t.id})[/bold]: {len(df)} cell types")
    tb = Table("CL id", "label", "relation", "via tissue", "census cells", "census datasets")
    for r in df.sort_values("label").itertuples():
        c = cx.get(r.cl_id)
        tb.add_row(r.cl_id, r.label, r.relation, r.tissue_label, f"{c[1]:,}" if c else "", str(c[2]) if c else "")
    rprint(tb)


@app.command("build-knowledge")
def build(
    tissue: str = typer.Option(..., help="Tissue name or UBERON id, e.g. 'brain' or UBERON:0000955"),
    panel: Optional[Path] = typer.Option(None, help="Gene panel: Xenium gene_panel.json, gene list file, or .h5ad"),
    out: Path = typer.Option(Path("knowledge"), help="Output directory"),
    species: str = "human",
    overrides: Optional[Path] = typer.Option(None, help="YAML with add/remove cell types, parents, marker edits"),
    sources: str = typer.Option("CellMarker2,PanglaoDB,ASCT+B,CellGuide", help="Comma-separated marker sources"),
    min_markers: int = 3,
    max_depth: int = 6,
    ancestor_levels: int = typer.Option(1, help="Also query cell types of parent tissues/systems (brain -> CNS)"),
    min_db_sources: int = typer.Option(2, help="DB-only cell types need this many sources annotating them to the tissue"),
    no_collapse: bool = typer.Option(False, help="Keep single-child non-seed nodes"),
    no_census: bool = typer.Option(False, help="Do not use CZ CELLxGENE Census observations as tissue evidence"),
    census_version: Optional[str] = typer.Option(None, help="Census release, e.g. 2025-01-30 (default: pinned LTS)"),
    census_min_cells: int = 200,
    census_min_datasets: int = 2,
    census_disease: str = typer.Option("normal", help="Census samples used as tissue evidence: 'normal' or 'any'"),
    force: bool = typer.Option(False, help="Re-download sources / re-query Ubergraph and Census"),
):
    """Build the tissue cell-type tree with panel-filtered markers and a coverage report."""
    from .knowledge.build import BuildParams, build_knowledge

    params = BuildParams(species=species, sources=tuple(s.strip() for s in sources.split(",") if s.strip()), ancestor_levels=ancestor_levels,
                         min_db_sources=min_db_sources, min_markers=min_markers, max_depth=max_depth, collapse_single_child=not no_collapse,
                         census=not no_census, census_min_cells=census_min_cells, census_min_datasets=census_min_datasets,
                         census_disease=census_disease,
                         **({"census_version": census_version} if census_version else {}))
    tree, cov, dropped = build_knowledge(tissue, panel=panel, params=params, overrides=overrides, out_dir=out, force=force)
    rprint(tree.render())
    rprint(f"\n[bold]{len(tree)} nodes, {len(tree.leaves())} leaves; {len(dropped)} dropped (see {out}/*_dropped.csv)")


@app.command("show-tree")
def show_tree(tree_json: Path, markers: int = 8, all_markers: bool = typer.Option(False, help="Show markers outside the panel too")):
    """Pretty-print a saved knowledge tree."""
    from .knowledge.tree import CellTypeTree

    t = CellTypeTree.load(tree_json)
    rprint(t.render(show_markers=markers, panel_only=not all_markers))
    rprint(t.meta)


@app.command("annotate")
def annotate(
    config: Path = typer.Option(..., help="Run config YAML (see configs/)"),
    dataset: Optional[str] = typer.Option(None, help="Only this dataset name from the config"),
    methods: str = typer.Option("hier,flat,rule,cluster", help="Comma-separated subset of hier,flat,rule,cluster"),
    subsample: Optional[int] = typer.Option(None, help="Random subsample of cells (for quick tests)"),
    force_knowledge: bool = typer.Option(False, help="Rebuild the knowledge tree even if cached"),
):
    """Load, preprocess and annotate the datasets in a config with the selected methods."""
    from .pipeline import load_config, run

    cfg = load_config(config)
    outs = run(cfg, datasets=[dataset] if dataset else None, methods=tuple(m.strip() for m in methods.split(",")), subsample=subsample,
               force_knowledge=force_knowledge)
    for k, v in outs.items():
        rprint(f"[green]{k}[/green] -> {v}")


@app.command("benchmark")
def benchmark(
    config: Path = typer.Option(..., help="Run config YAML with a 'benchmark' section (reference_column, reference_map)"),
    dataset: str = typer.Option(..., help="Dataset name from the config"),
    methods: str = typer.Option("hier,flat,rule,cluster", help="Comma-separated subset of hier,flat,rule,cluster"),
    subsample: Optional[int] = typer.Option(None, help="Random subsample of cells (forces a fresh annotation run)"),
    no_reuse: bool = typer.Option(False, help="Re-annotate even if results/<dataset>/annotated.h5ad exists"),
    force_knowledge: bool = typer.Option(False, help="Rebuild the knowledge tree even if cached"),
):
    """Compare hier/flat/cluster annotations with an expert reference column; writes tables, figures and report.md."""
    from .benchmark import run_benchmark
    from .pipeline import load_config

    cfg = load_config(config)
    out = run_benchmark(cfg, dataset, methods=tuple(m.strip() for m in methods.split(",")), subsample=subsample, reuse=not no_reuse,
                        force_knowledge=force_knowledge)
    import pandas as pd

    summ = pd.read_csv(out / "summary.csv", index_col=0)
    cols = [c for c in ("lineage_acc", "exact_acc", "coarser", "unassigned", "wrong", "hF1", "ARI_assigned", "spatial_coherence", "runtime_sec") if c in summ]
    rprint(summ[cols].round(3).to_string())
    rprint(f"[green]report[/green] -> {out / 'report.md'}")


if __name__ == "__main__":  # pragma: no cover
    app()
