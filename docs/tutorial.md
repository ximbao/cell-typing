# Tutorial: annotating an imaging-based spatial transcriptomics dataset

This walkthrough annotates a **10x Xenium Prime 5K ovarian carcinoma FFPE** section (407k cells, 5,101 genes)
with `celltyping`. Everything shown applies unchanged to any Xenium or CosMx dataset: only the input path, the
tissue name and (optionally) the override file change.

```python
import celltyping as ct
import scanpy as sc
```

## 1. Load the data

`celltyping` works on an `AnnData` with **raw counts in `X`** and spatial coordinates in `obsm["spatial"]`.
Use one of the readers or your own `.h5ad`:

```python
# Xenium output folder (cell_feature_matrix.h5 + cells.parquet or cells.csv.gz)
adata = ct.read.xenium("/path/to/Xenium_outs")

# CosMx flat files (exprMat + metadata csv)
adata = ct.read.cosmx("/path/to/CosMx_flatfiles")

# an existing h5ad (this is what we use for the ovarian dataset)
adata = ct.read.h5ad("ov_validation.h5ad")
adata
# AnnData object with n_obs x n_vars = 407124 x 5101
#     obs: 'cell_id', 'transcript_counts', ..., 'cell_area', 'cell_type'
#     obsm: 'spatial'
```

The readers move control probes (negative controls, blanks, unassigned codewords) out of the gene matrix into
`obs["control_counts"]`, so `X` contains genes only.

## 2. One call: `ct.annotate`

```python
ct.annotate(adata, tissue="ovary")
```

That single call runs the whole workflow and writes the result into `adata` (nothing is removed, the object is
also returned):

```
INFO X looks like raw counts -> preprocessing on
INFO QC: 384278 / 407124 cells high quality (22846 flagged low quality: low_counts=22846)
INFO kNN-smoothed expression (k=15, 384278 high-quality cells) stored in layers['knn_smooth']
INFO tissue: ovary (UBERON:0000992)
INFO panel: 5101 genes
INFO Census (2025-01-30, tissue_general=ovary, disease=normal): 44 cell types observed, 10 pass ...
INFO expected cell types: 39 from ontology (39 direct), 11 from >=2 marker DBs/Census, 10 from Census, 50 total
INFO CellGuide: markers for 21 / 47 cell types (19 with ovary-specific entries)
INFO tree: 23 nodes, 16 leaves, max depth 3; 58 nodes dropped
INFO hierarchical annotation: 3.4% unassigned at root; 23 distinct labels; top: malignant cell=..., fibroblast=...
INFO 22846 low-quality cells labelled 'low_quality'
INFO annotated 407124 cells (22846 low quality) in 150s -> obs['hier_label'] (24 types; malignant cell 41%, ...)
```

What happened, step by step (each step is also a public function, see below):

| step | what it does | where the result lives |
|------|--------------|------------------------|
| gene harmonisation | upper-cases gene symbols so they match the marker databases | `var["original_symbol"]` keeps the originals |
| **QC flag** | cells with `transcript_counts < 20` are flagged `low_quality`; they are kept but excluded from every later step | `obs["qc_flag"]`, `obs["qc_pass"]`, `obs["qc_reason"]`, `uns["qc"]` |
| normalisation | `normalize_total` + `log1p`; raw counts kept | `X` (log-normalised), `layers["counts"]` |
| kNN smoothing | each high-quality cell's expression is averaged with its 15 nearest expression neighbours (Xenium/CosMx counts are sparse; a cell shows only a few of its marker genes) | `layers["knn_smooth"]` |
| knowledge | tissue -> expected cell types (Cell Ontology / Uberon, CZ CELLxGENE Census) -> marker hierarchy restricted to the genes of *this* panel; cached, so the next dataset with the same panel reuses it | `uns["celltyping"]["tree"]` |
| hierarchical annotation | cells descend the tree top-down; a cell stops at a coarser label when its children are not clearly separable | `obs["hier_label"]` and friends |

The label of interest is `obs["hier_label"]`:

```python
adata.obs["hier_label"].value_counts()
# malignant cell        168k
# fibroblast             63k
# smooth muscle cell     36k
# low_quality            23k
# pericyte               22k
# macrophage             21k
# endothelial cell       21k
# unassigned             13k
# stromal cell           11k
# ...
```

Three special labels can appear: `low_quality` (failed QC), `unassigned` (no cell type scored above the
thresholds at the root) and internal-node labels such as `epithelial cell` or `leukocyte` (the cell reached that
lineage but no child won clearly). All columns are documented in [annotate_output.md](annotate_output.md).

### Useful options

```python
ct.annotate(adata, tissue="ovary",
            species="human",                    # or "mouse"
            min_counts=20,                      # QC threshold (transcripts per cell)
            knn_smooth=15,                      # neighbours for expression smoothing; None disables
            overrides="configs/overrides_ovary.yaml",   # curated edits to the tree (see below)
            method="hierarchical",              # or "flat" / "cluster"
            min_score=0.5, min_margin=0.25,     # stricter -> more 'unassigned' / coarser labels
            knowledge_kwargs={"census_disease": "any", "min_markers": 5})
```

`tissue` accepts any Uberon term name or id. `celltyping search-tissue "fallopian"` on the command line (or
`ct.tl.expected_cell_types("ovary")`) helps to pick the right one and to see what the ontology and the Census expect
in that tissue *before* running anything.

## 3. Look at the result

```python
import matplotlib.pyplot as plt

tree = ct.tl.get_tree(adata)              # the tree used for this run
ct.pl.tree(tree)                          # hierarchy with panel-marker counts, Census cell counts, top markers
ct.pl.spatial(adata, "hier_label", tree=tree)
ct.pl.composition(adata)                  # composition per hierarchy level
ct.pl.scores(adata, tree=tree)            # mean signature score per predicted label x cell type
ct.pl.markers(adata, tree)                # scanpy dot-plot of each type's top markers
ct.pl.spatial(adata, "hier_score")        # confidence (numeric obs columns are supported)
plt.show()
```

`ct.pl.scores` is the first thing to check: every predicted type should light up on its own signature (the
diagonal); off-diagonal blocks show which types are hard to separate with this panel.

`print(tree.render())` prints the hierarchy as text; `tree.save("ovary_tree.json")` stores it for reuse
(`ct.annotate(adata, tree="ovary_tree.json")`).

## 4. Step by step (scanpy-style)

The same workflow with each step explicit, which is what you want when tuning:

```python
adata = ct.read.h5ad("ov_validation.h5ad")

# preprocessing
ct.pp.harmonize_genes(adata)
ct.pp.qc(adata, min_counts=20)             # adds obs['qc_flag']; use filter=True to drop low-quality cells instead
ct.pp.normalize(adata)                     # log-normalise, counts -> layers['counts']
ct.pp.knn_smooth(adata, k=15)              # layers['knn_smooth'] (high-quality cells only)
#   or all of the above: adata = ct.pp.preprocess(adata, min_counts=20, knn_smooth_k=15)

# knowledge: which cell types to expect and which of their markers are on the panel
ct.tl.expected_cell_types("ovary")         # table: CL id, label, ontology relation, Census cells/datasets
tree = ct.tl.knowledge("ovary", panel=adata, overrides="configs/overrides_ovary.yaml")
print(tree.render())

# annotators (all three read layers['knn_smooth'] when present and skip low-quality cells)
ct.tl.hierarchical(adata, tree)                        # obs['hier_*']
ct.tl.flat(adata, tree)                                # obs['flat_*']   prior-knowledge baseline, all leaves at once
ct.tl.clusters(adata, tree, resolutions=(0.5, 1.0))    # obs['cluster_r0.5_*'] Leiden + DE + marker overlap

# save
adata.write_h5ad("ov_annotated.h5ad")
```

### Inspecting and fixing the knowledge

`ct.tl.knowledge` writes a coverage table next to the cached tree (`ct.settings.knowledge_dir`) and can also write
it to a folder of your choice:

```python
tree = ct.tl.knowledge("ovary", panel=adata, out_dir="results/ovary_knowledge")
# results/ovary_knowledge/ovary_tree.json, ovary_tree.txt, ovary_coverage.csv, ovary_dropped.csv
```

`ovary_coverage.csv` lists, per cell type, how many of its markers are on the panel, which sources support them,
and the Census cell/dataset counts; `ovary_dropped.csv` lists the cell types that were expected but had too few
panel markers to be scored. Typical findings and the matching fix in an **override YAML**:

```yaml
# configs/overrides_ovary.yaml (excerpt)
add_cell_types:                       # types the ontology does not place in the tissue (label comes from CL)
  - {id: "CL:0001064", parent: "CL:0000000"}          # malignant cell, directly under the root
  - {id: "CL:4030007", parent: "CL:4052018"}          # fallopian tube multiciliated cell under tubal epithelium
remove_cell_types:                    # expected by the ontology, absent from this sample
  - "CL:0000586"                      # germ cell lineage: not in a carcinoma section
collapse_cell_types:                  # keep the children, drop the intermediate
  - "CL:0000763"                      # myeloid cell
parents:                              # force a parent where the ontology's is_a is not what you want
  "CL:0000669": "CL:0000192"          # pericyte under smooth muscle cell
markers:
  "CL:0001064": {replace: [PAX8, WT1, MUC16, MSLN, EPCAM, ...]}   # own marker list for the tumour compartment
  "CL:0000057": {add: [FAP, POSTN], remove: [ACTA2]}
```

Rebuild with `ct.tl.knowledge("ovary", panel=adata, overrides="my_overrides.yaml")` and re-run the annotator; the
cache key includes the override file, so nothing stale is reused.

### Tuning the annotation

- `min_score` / `min_margin` (defaults 0.5 / 0.25, in null-SD units): raising them produces more `unassigned` and
  more cells stopping at coarser labels; lowering them forces more leaf calls.
- `knn_smooth`: 10-20 works for Xenium/CosMx; set `None` for very dense panels or if you want strictly per-cell
  calls.
- `subtree_mode="max"` in `ct.tl.hierarchical` for heterogeneous lineages (e.g. an `epithelial cell` node that
  groups tumour, tubal, urothelial and mesothelial cells): each member is scored separately instead of as a union.
- `smooth={"k": 10, "min_frac": 0.6}` applies a spatial majority vote to the final labels.
- Compare methods on your own data with `ct.bm.compare` if you have reference labels: see
  [benchmarking.md](benchmarking.md).

## 5. Command line

The same workflow is available as a CLI driven by a YAML config (`configs/ovarian_xenium.yaml` is the one used
here):

```bash
celltyping search-tissue ovary
celltyping expected-cell-types ovary                        # ontology + Census expectation, before markers
celltyping build-knowledge --tissue ovary --panel gene_panel.json \
    --overrides configs/overrides_ovary.yaml --out results/ovary/knowledge
celltyping show-tree results/ovary/knowledge/ovary_tree.json
celltyping annotate --config configs/ovarian_xenium.yaml --dataset ov_validation --methods hier,flat,cluster
```

`celltyping annotate` writes `results/<run>/<dataset>/annotated.h5ad`, `labels.csv` (all label columns, the QC
flag and coordinates) and the cluster-labelling tables.

## 6. Settings and caches

```python
ct.settings.data_dir = "/shared/celltyping_cache"   # ontologies, marker DBs, Census/CellGuide tables (~1 GB)
ct.settings.knowledge_dir = "/shared/trees"          # built trees (default: data_dir/knowledge)
ct.settings.species = "human"
ct.settings.census = True                            # use CZ CELLxGENE Census observations as tissue evidence
ct.settings.verbosity = 1                            # 0 warnings, 1 info, 2 debug
```

The first run for a new tissue needs network access (Ubergraph, marker databases, Census); subsequent runs are
offline. Without the optional `cellxgene-census` package the Census step is skipped with a warning.
