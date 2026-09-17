# celltyping

Ontology-driven, hierarchical cell typing for imaging-based spatial transcriptomics (10x Xenium, NanoString CosMx),
plus the two conventional strategies (unsupervised clustering + marker labelling; flat signature scoring) built on the
same knowledge and preprocessing so they can be compared fairly.

## Idea

Prior-knowledge cell typing needs a marker list for the tissue at hand. Instead of hand-writing it:

1. the user names a **tissue** (`ovary`, `prostate gland`, `UBERON:0002048`, ...);
2. **Uberon / Cell Ontology** (via Ubergraph, `part_of`/`located_in` with reasoning) give the cell types expected in
   that tissue and their `is_a` hierarchy;
3. **CZ CELLxGENE Census** adds the cell types actually *observed* in that tissue by expert-annotated single-cell
   datasets (with cell and dataset counts as evidence);
4. **marker databases** (CellMarker 2.0, PanglaoDB, HuBMAP ASCT+B, CELLxGENE CellGuide computational markers) are
   joined on Cell Ontology ids and filtered to the **gene panel** of the experiment, with a coverage report;
5. cells are classified **top-down** through the tree: at each node only the cells assigned to that node are scored
   against its children; a cell descends only when the best child clearly wins, otherwise it keeps the coarser label.

## Install

```bash
mamba env create -f environment.yml
mamba activate celltyping
pip install -e .
```

Downloads (ontologies, marker tables, Ubergraph query results, Census/CellGuide tables) are cached under `data/`
(override with `celltyping.settings.data_dir = ...` or `CELLTYPING_DATA`). The Census tissue evidence needs the optional
`cellxgene-census` package (`pip install -e ".[census]"`, included in `environment.yml`); without it the build falls
back to ontology + marker databases with a warning.

## Quick start

```python
import celltyping as ct

adata = ct.read.xenium("/path/to/xenium/outs")          # or ct.read.cosmx(...), ct.read.h5ad(...)
ct.annotate(adata, tissue="ovary")                       # QC flag + normalise + smooth, build & cache the ovary tree
                                                         # for this panel, hierarchical annotation -- all in place
adata.obs["hier_label"].value_counts()                   # 'low_quality' = failed QC, 'unassigned' = no type called
ct.pl.spatial(adata, "hier_label")
```

Cells with fewer than 20 transcripts (`min_counts`) are flagged `obs["qc_flag"] == "low_quality"`, kept in the
object, and excluded from smoothing, scoring and benchmarking. See the [tutorial](docs/tutorial.md), the
[output reference](docs/annotate_output.md) and the [benchmarking guide](docs/benchmarking.md).

`ct.annotate` is the one-call path; every step is also available scanpy-style:

```python
ct.settings.species = "human"                            # defaults; also data_dir, knowledge_dir, census, sources, verbosity

ct.pp.preprocess(adata)                                  # = pp.harmonize_genes + pp.qc (flag) + pp.normalize + pp.knn_smooth
tree = ct.tl.knowledge("ovary", panel=adata, overrides="configs/overrides_ovary.yaml")   # cached per tissue/panel/params
ct.tl.expected_cell_types("ovary")                       # ontology + Census evidence table, before marker filtering

ct.tl.hierarchical(adata, tree)                          # obs: hier_label, hier_id, hier_level1.., hier_score, hier_margin
ct.tl.flat(adata, tree)                                  # obs: flat_label ...          (prior-knowledge baseline)
ct.tl.clusters(adata, tree, resolutions=(0.5, 1.0))      # obs: cluster_r0.5_label ...  (clustering baseline)
ct.tl.score(adata, tree, nodes=["CL:0000057"])           # raw signature scores for any nodes

ct.pl.tree(tree); ct.pl.composition(adata); ct.pl.scores(adata, tree=tree); ct.pl.markers(adata, tree)

ref = ct.bm.reference(adata, "cell_type", {"Tumor Cells": "CL:0001064", ...}, tree)   # expert labels -> tree nodes
ct.bm.compare(adata, ref, tree, methods=("hier", "flat", "cluster"))                   # one metrics row per method
ct.pl.confusion(ct.bm.confusion(adata, ref, tree, key="hier"))
```

## CLI

```bash
celltyping search-tissue "prostate"                        # find the right Uberon term
celltyping expected-cell-types ovary                       # what the ontology and the Census put in the tissue
celltyping build-knowledge --tissue ovary \
    --panel /path/to/gene_panel.json --overrides configs/overrides_ovary.yaml --out results/ovary/knowledge \
    [--no-census] [--census-version 2025-01-30] [--sources CellMarker2,PanglaoDB,ASCT+B,CellGuide]
celltyping show-tree results/ovary/knowledge/ovary_tree.json
celltyping annotate --config configs/ovarian_xenium.yaml --dataset ov_validation --methods hier,flat,cluster [--subsample 20000]
celltyping benchmark --config configs/ovarian_xenium.yaml --dataset ov_validation   # vs expert obs['cell_type']
```

`build-knowledge` writes `<tissue>_tree.json` (the classifier input), `<tissue>_coverage.csv` (markers per node in the
panel, sources), `<tissue>_dropped.csv` (cell types removed for lack of panel markers) and `<tissue>_tree.txt`.
Edit `configs/overrides_*.yaml` to add/remove cell types, force parents, or add/remove markers, then rebuild.

## Lower-level API

The facade wraps these modules, which remain importable: `celltyping.knowledge.build.build_knowledge`,
`celltyping.io.read_xenium/read_cosmx`, `celltyping.preprocess`, `celltyping.methods.{annotate_hierarchical,
annotate_flat, annotate_clusters}`, `celltyping.benchmark.*`.

## Layout

- `src/celltyping/__init__.py`, `_annotate.py`, `settings.py`, `read.py`, `pp.py`, `tl.py`, `pl.py`, `bm.py` the scanpy-style facade
- `src/celltyping/knowledge/` ontology access (`ontology.py`), marker sources (`markers.py`), CZ CELLxGENE Census +
  CellGuide (`cellxgene.py`), tree (`tree.py`), builder (`build.py`)
- `src/celltyping/io.py`, `preprocess.py` readers and shared QC/normalisation
- `src/celltyping/methods/` `hierarchical.py`, `flat.py`, `cluster.py`, shared `scoring.py`
- `src/celltyping/pipeline.py`, `cli.py` config-driven runs
- `src/celltyping/benchmark/` reference harmonisation (`harmonize.py`), metrics, figures/report, driver (`run.py`)
- `configs/` run config and override file for the ovarian example (`ovarian_xenium.yaml`, `overrides_ovary.yaml`; `overrides_brain.yaml` is a second curation example)
- `docs/` [tutorial](docs/tutorial.md), [benchmarking guide](docs/benchmarking.md), [output reference](docs/annotate_output.md)
- `tests/` unit tests on synthetic data (no network needed)

## Knowledge sources

- **Ontology**: Ubergraph `part_of` / `located_in` (+ reasoning) from the tissue and its parent tissues
  (`ancestor_levels`), `is_a` closure to build the tree; ancestor-tissue hits need corroboration by a marker database
  or the Census.
- **CZ CELLxGENE Census** (`knowledge.census`, default on): `cellxgene_census.get_obs` for the Census
  `tissue_general` category containing the tissue (mapped through Uberon; e.g. `cerebral cortex -> brain`), primary
  cells of healthy samples (`census_disease: normal`; `any` includes disease datasets), aggregated per Cell Ontology
  term (cells, datasets; cached per Census release, one query of 15-30 s per tissue). A type observed with
  `>= census_min_cells` cells, in `>= census_min_datasets` datasets and making up `>= census_min_frac` of the
  tissue's cells seeds the tree and counts as one corroborating source; Census-only seeds that are `is_a` ancestors
  of other seeds (`neural cell`, `macroglial cell`) stay structural so they do not add hierarchy levels. The counts
  are kept on the nodes (`census_n_cells` in the coverage report, shown by `ct.pl.tree`). On the ovary panel this
  adds `stromal cell`; with `census_disease: any` also `monocyte` (tumour/ascites datasets). On large Census tissues
  (brain: 20M cells) the fraction threshold is what keeps blood contaminants out.
- **Marker databases**: CellMarker 2.0, PanglaoDB, HuBMAP ASCT+B (curated), and CellGuide's *computational* markers
  (per organism and `tissue_general`, filtered by specificity >= 0.7 and detection in >= 10% of the type; weight =
  absolute `marker_score`, clipped at 2). Computational markers are attached to seed nodes only and do **not** count
  towards `min_markers`: they refine the signature of types the curated sources establish, but cannot keep a type
  alive on their own (CellGuide has genes for almost every CL term, and e.g. its ovary `theca cell` list is a generic
  stromal programme that would otherwise absorb tumour fibroblasts). Weights from different sources add up.
- Non-seed nodes that end up without descendants (dead-end `is_a` ancestors) are dropped.

## Scoring notes

- Default score is `robust_z`: per gene, expression is mean-centred over cells (`scoring.scale: center`); per cell, the
  mean of the top fraction (`top_frac`) of a set's weighted values is calibrated against random gene sets of the same
  size, so scores are in null-SD units and comparable across marker sets of different size. `scale: z` gives classic
  per-gene z-scores (with an SD floor); on sparse imaging data they let rarely detected genes dominate a set from a
  single count and performed clearly worse on the ovarian benchmark (root-level accuracy 0.66 -> 0.80 for centring).
  `mean_z`, and `ulm`/`aucell` via `decoupler`, are available with `annotate.method`.
- QC does not remove cells: `pp.qc` flags cells with `total_counts < min_counts` (default 20) as `low_quality`
  (`obs['qc_flag']`, `obs['qc_pass']`); smoothing neighbours are searched among high-quality cells only, the
  annotators label low-quality cells `low_quality` and the benchmark excludes them. `filter=True` removes them instead.
- Xenium/CosMx counts are sparse; `preprocess.knn_smooth: k` averages each cell with its k expression neighbours
  (stored as a layer) before scoring. Set `knn_smooth: 0` to score raw log-normalised counts.
- In the hierarchy, internal nodes are scored with **subtree signatures** (descendants' markers merged, sibling-shared
  genes down-weighted; structural non-seed nodes such as `epithelial cell` contribute no pan-tissue database rows of
  their own) and the parent's own marker set competes with its children (`parent_as_competitor`), so a cell descends
  only when a child clearly wins over "stay here". `subtree_mode: max` scores each subtree member separately and takes
  the per-cell maximum, for heterogeneous subtrees.
- Override files accept `markers: {<CL id>: {replace: [...]}}` to discard database rows for a node entirely (used for
  `malignant cell`, whose database markers are pan-cancer).

## Benchmark

`celltyping benchmark --config configs/ovarian_xenium.yaml --dataset ov_validation` runs (or reuses) the three
annotators and compares each with an expert annotation column. Expert labels are mapped onto Cell Ontology nodes of the
tree in the config (`benchmark.reference_map`); a prediction counts as correct at the reference's granularity when it is
the reference node or one of its descendants (`lineage_acc`), as `coarser` when it stops at an ancestor, otherwise
`wrong` or `unassigned`. Also reported: hierarchical P/R/F1, ARI/NMI on projected labels, per-class precision/recall,
spatial coherence (k nearest spatial neighbours sharing the label), marker specificity and runtime. Outputs go to
`results/<name>/<dataset>/benchmark/` (`summary.csv`, `per_class_*.csv`, `confusion_*.{csv,png}`, `spatial_labels.png`,
`report.md`).

### Results: Xenium Prime 5K ovarian carcinoma FFPE (`configs/ovarian_xenium.yaml`)

407k cells, expert `obs['cell_type']` (18 labels, 6 of them tumour states) mapped onto 10 nodes of an ovary tree built
from CL/Uberon + Census (healthy ovary) + CellMarker/PanglaoDB/ASCT+B/CellGuide with `configs/overrides_ovary.yaml`
(adds the malignant compartment and generic stroma/immune types, removes germ-line / follicle terms). 22 nodes with
panel markers; 382k evaluated high-quality cells (23k of 407k cells flagged `low_quality`); full tables in `results/ovarian_xenium/ov_validation/benchmark/report.md`.

| method        | lineage_acc | exact | coarser | unassigned | wrong | hF1  | macro-F1 | ARI  | spatial coherence | labels | runtime |
|---------------|------------:|------:|--------:|-----------:|------:|-----:|---------:|-----:|------------------:|-------:|--------:|
| expert        |             |       |         |            |       |      |          |      | 0.63              | 10     |         |
| hierarchical  | 0.79        | 0.74  | 0.02    | 0.04       | 0.15  | 0.79 | 0.48     | 0.68 | 0.57              | 22     | 19 s    |
| flat (leaves) | 0.81        | 0.66  | 0.00    | 0.03       | 0.17  | 0.74 | 0.53     | 0.64 | 0.59              | 16     | 18 s    |
| cluster r=0.5 | 0.83        | 0.69  | 0.00    | 0.00       | 0.17  | 0.76 | 0.42     | 0.72 | 0.65              | 7      | 139 s   |
| cluster r=1.0 | 0.77        | 0.68  | 0.00    | 0.01       | 0.22  | 0.73 | 0.41     | 0.69 | 0.61              | 9      | 139 s   |

Per-class F1 (hier / flat / cluster r=0.5): tumour 0.87 / 0.85 / 0.91, fibroblast 0.85 / 0.84 / 0.87, smooth muscle
0.87 / 0.84 / 0.90, endothelium 0.88 / 0.88 / 0.72, macrophage 0.66 / 0.73 / 0.60, lymphocyte 0.38 / 0.55 / 0.16,
ciliated tubal cells 0.13 / 0.29 / 0, urothelial-like 0.01 / 0.23 / 0, pericyte 0.15 / 0.06 / 0.01.

- Clustering wins on the bulk classes because whole clusters are labelled at once, but it only produces 7-10 labels:
  it never finds fallopian tube epithelium, ciliated or urothelial-like cells, calls the smooth-muscle cluster
  `pericyte` and mixes lymphocytes with macrophages.
- The two per-cell methods are complementary: the hierarchical annotator has the highest exact-label agreement (0.74)
  and hF1 (0.79) and resolves smooth muscle vs pericyte and the stroma/tumour boundary better, while flat leaf scoring
  with the same markers finds more of the rare epithelial and lymphoid types (the hierarchy stops many of them at
  `epithelial cell` / `leukocyte`, which is counted as `coarser`/`wrong` at the reference's granularity).
- Remaining errors are informative: fallopian tube / urothelial-like epithelium is called `malignant cell` (HGSC and
  tubal secretory cells share PAX8/WT1/EPCAM/MUC16; not separable by markers alone), `luteal cell` (an
  ontology-expected normal-ovary type) attracts a few percent of stromal cells, and lymphocytes (small cells, few
  transcripts) often stop at `leukocyte`. These are curation levers in the override file, not scoring problems.

Effect of the knowledge sources on the same data (hierarchical / flat lineage accuracy, hF1): ontology + curated
databases only 0.78 / 0.74 (hF1 0.78 / 0.69); + Census (healthy) seeds + CellGuide markers **0.79 / 0.81 (0.79 / 0.74)**;
+ Census with disease datasets (`census_disease: any`, adds `monocyte`) 0.77 / 0.79 (0.77 / 0.73); CellGuide markers
allowed to keep nodes alive (adds `theca cell`, `stromal cell of ovary`) 0.71-0.74 / 0.65-0.71, because their generic
stromal programme absorbs a quarter of the tumour fibroblasts -- the reason computational markers do not count towards
`min_markers`.
