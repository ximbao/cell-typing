# `celltyping.annotate`: parameters and output

```python
ct.annotate(adata, tissue=None, species=None, tree=None, method="hierarchical", overrides=None,
            preprocess="auto", min_counts=20, max_control_frac=0.3, knn_smooth=15, min_score=0.5, min_margin=0.25, top_k=30,
            key=None, knowledge_kwargs=None, copy=False, **method_kwargs)
```

The function modifies `adata` in place (and returns it). No cells are removed: cells failing QC are flagged and
labelled `low_quality`.

## Parameters

| parameter | default | meaning |
|-----------|---------|---------|
| `adata` | | `AnnData`, cells x genes. `X` may hold raw counts (preprocessing runs) or log-normalised values (preprocessing is skipped). `obsm["spatial"]` is optional (used by plots, label smoothing and the benchmark's spatial coherence). |
| `tissue` | `None` | Uberon term name or id (`"ovary"`, `"cerebral cortex"`, `"UBERON:0002048"`). Required unless `tree` is given. Resolves through Uberon synonyms; ambiguous names are logged with the alternatives (`celltyping search-tissue <name>` lists them). |
| `species` | `settings.species` (`"human"`) | `"human"` or `"mouse"`. Selects marker-database species, CellGuide organism and the Census organism, and removes the other species' provisional CL terms. |
| `tree` | `None` | A prebuilt `CellTypeTree` or a path to a saved `*_tree.json`; skips the knowledge build. |
| `method` | `"hierarchical"` | `"hierarchical"` (top-down through the tree), `"flat"` (all leaves scored at once, prior-knowledge baseline), `"cluster"` (Leiden + DE + marker overlap, clustering baseline), `"rule_based"` (fixed panels: all genes must have count > 0; winner = highest raw-count sum). |
| `overrides` | `None` | YAML file (or `Overrides` object) with curated tree edits: `add_cell_types`, `remove_cell_types`, `collapse_cell_types`, `parents`, `markers` (`add`/`remove`/`replace` per CL id), `custom_markers` (CSV). |
| `preprocess` | `"auto"` | `"auto"`: normalise only if `X` looks like raw integer counts; `True`/`False` force it. Gene harmonisation and the QC flag are always applied when missing. |
| `min_counts` | `20` | QC threshold: cells with `transcript_counts < min_counts` are flagged `low_quality` and excluded from smoothing, scoring and the benchmark. Uses `obs['transcript_counts']` (Xenium metadata when present, otherwise the sum of `X` / `layers['counts']`). Xenium's `total_counts` (gene + codewords) is not used. |
| `max_control_frac` | `0.3` | Negative-control fraction threshold (`qc_reason='high_control_frac'`). `None` disables. |
| `knn_smooth` | `15` | Number of expression neighbours (PCA space, high-quality cells only) whose log-expression is averaged with each cell before scoring; stored in `layers["knn_smooth"]`. `None`/`0` disables smoothing. Ignored if the layer already exists. **Disabled for `rule_based`** (uses raw counts). |
| `min_score` | `0.5` | Minimum signature score (in null-SD units, see below) for a cell type to be called. Below it a cell is `unassigned` (flat) or stays at the current node (hierarchical). |
| `min_margin` | `0.25` | Minimum difference between the best and the second-best candidate (and, in the hierarchy, between the best child and the parent's own signature). Controls how readily cells descend to leaves. |
| `top_k` | `30` | Markers per cell type used in the signature (highest weights first). `None` uses all panel markers. |
| `key` | method-dependent | Prefix of the result columns: `"hier"`, `"flat"`, `"rule"` or `"cluster"`. |
| `knowledge_kwargs` | `None` | Extra arguments for `ct.tl.knowledge`: `min_markers` (default 5), `max_depth` (5), `ancestor_levels` (1), `min_db_sources` (2), `census`, `census_disease` (`"normal"`/`"any"`), `census_min_cells`, `census_min_datasets`, `census_min_frac`, `sources`, `force`, `cache`, `out_dir`. |
| `copy` | `False` | Work on a copy and return it. |
| `**method_kwargs` | | Forwarded to `ct.tl.hierarchical` / `ct.tl.flat` / `ct.tl.clusters` / `ct.tl.rule_based`, e.g. `subtree_mode="max"`, `parent_as_competitor=False`, `smooth={"k": 10, "min_frac": 0.6}`, `scale="z"`, `top_frac=0.34`, `level="all"`, `resolutions=(0.5, 1.0)`, `n_markers=4`, `markers_dict={...}`. |

### How a score is computed

Every candidate cell type has a weighted marker set (weights add up over the sources that list a gene:
CellMarker 2.0, PanglaoDB, ASCT+B, CellGuide, overrides; genes shared between sibling types are down-weighted).
For each cell, expression (smoothed layer if present) is mean-centred per gene; the score is the mean of the
top `top_frac` (default 50 %) of the set's weighted values, calibrated against random gene sets of the same size,
so that **scores are in null standard deviations** and comparable across cell types with different numbers of
markers. A score of 0.5 means "half a standard deviation above what a random gene set of this size gives".

In the hierarchy, a node with descendants is scored by the merged markers of its whole subtree
(`subtree_signatures=True`), so a coarse decision (e.g. epithelial vs stromal vs immune) uses the
well-characterised leaf markers.

## Preprocessing output

| location | added by | content |
|----------|----------|---------|
| `var["original_symbol"]` | harmonisation | gene symbols as they were in the input; `var_names` are upper-cased |
| `obs["transcript_counts"]` | QC | gene transcripts per cell; compared against `min_counts` |
| `obs["n_genes_by_counts"]` | QC | detected genes per cell |
| `obs["control_frac"]` | QC | negative-control fraction: `(control_probe_counts + genomic_control_counts) / (transcript_counts + controls)` on Xenium; flagged when `> max_control_frac` (default **0.3**) |
| `obs["qc_pass"]` | QC | `True` for high-quality cells |
| `obs["qc_flag"]` | QC | categorical `high_quality` / `low_quality` |
| `uns["qc"]` | QC | thresholds, number of low-quality cells and the reason counts (`low_counts`, `few_genes`, `control_probes`) |
| `layers["counts"]` | normalisation | raw counts (copied before `normalize_total` + `log1p` overwrote `X`) |
| `uns["normalized"]` | normalisation | `{"target_sum": ..., "log1p": True}`; also the signal that `X` is no longer raw |
| `layers["knn_smooth"]` | smoothing | kNN-averaged log-expression of high-quality cells (low-quality rows are copies of `X`) |
| `obsm["X_pca"]` | smoothing | PCA used for the neighbour search (NaN rows for low-quality cells) |
| `uns["knn_smooth"]` | smoothing | `{"k": 15, "layer": "knn_smooth", "n_cells_smoothed": ...}` |

## Annotation output: `method="hierarchical"` (prefix `hier`)

| column | type | meaning |
|--------|------|---------|
| `obs["hier_label"]` | categorical | **final cell-type label**: the deepest node the cell reached. Leaf names (`fibroblast`), internal nodes when the descent stopped (`epithelial cell`, `leukocyte`), `unassigned` (nothing scored above `min_score` at the root) or `low_quality`. |
| `obs["hier_id"]` | str | Cell Ontology id of `hier_label` (`CL:0000057`). `CL:0000000` (root) for unassigned cells, `low_quality` for flagged cells. Use this column for ontology-aware comparisons. |
| `obs["hier_level1"]`, `obs["hier_level2"]`, ... | categorical | the label at each depth of the hierarchy. `hier_level1` is the lineage (e.g. `epithelial cell`, `stromal cell`, `leukocyte`, `endothelial cell`). A cell that stopped at depth *d* repeats its label in all deeper levels, so every level is a complete labelling at that granularity. |
| `obs["hier_depth"]` | int / NaN | depth of the final node (0 = unassigned at root). NaN for low-quality cells. |
| `obs["hier_score"]` | float | signature score of the final node for this cell, in null-SD units (NaN if unassigned / low quality). |
| `obs["hier_margin"]` | float | score difference between the winning node and its strongest competitor at the last decision (higher = more confident). |
| `obsm["hier_scores"]` | DataFrame (cells x cell types) | the full score table; NaN where a node was not evaluated for a cell (it was not on the cell's path) or the cell is low quality. Columns are cell-type labels. |
| `uns["hier_params"]` | dict | all parameters of the run and the tree metadata (tissue, species, sources, panel size). |
| `obs["hier_label_smooth"]` | categorical | only with `smooth={...}`: labels after spatial majority vote. |

## Annotation output: `method="flat"` (prefix `flat`)

| column | meaning |
|--------|---------|
| `obs["flat_label"]`, `obs["flat_id"]` | best-scoring leaf (or `unassigned` / `low_quality`) |
| `obs["flat_score"]`, `obs["flat_margin"]` | its score and the gap to the runner-up |
| `obsm["flat_scores"]` | scores of all candidates for all cells |
| `uns["flat_params"]` | parameters, incl. `level` (`"leaves"`, `"all"` or an integer depth) and `n_candidates` |


## Annotation output: `method="rule_based"` (prefix `rule`)

Panel rule: for each cell type, the top `n_markers` genes from the knowledge tree (or a custom `markers_dict`) form a
panel. A cell **qualifies** for a panel only if **every** gene in the panel has raw count > 0. Among qualifying panels,
the winner is the one with the **highest sum of raw counts**; ties break by panel order. Cells with no qualifying panel
are `unassigned`. Uses `layers['counts']` when present (after preprocessing), otherwise `X`. kNN smoothing is not applied.

| column | meaning |
|--------|---------|
| `obs["rule_label"]`, `obs["rule_id"]` | winning panel label / CL id (`unassigned` / `low_quality`) |
| `obs["rule_score"]` | sum of raw counts for the winning panel (NaN if unassigned) |
| `obsm["rule_scores"]` | raw count sums per panel (0 where the panel did not qualify) |
| `uns["rule_params"]` | parameters, panel gene lists and the rule description |

## Annotation output: `method="cluster"` (prefix `cluster`)

For each Leiden resolution *r* (default 0.5 and 1.0), with `R = f"cluster_r{r:g}"`:

| column | meaning |
|--------|---------|
| `obs[R]` | Leiden cluster id |
| `obs[R + "_label"]`, `obs[R + "_id"]` | cell type assigned to the cluster (all cells of a cluster share it) |
| `obsm[R + "_type_scores"]` | per-cell copy of the cluster's cell-type scores |
| `uns[R + "_de"]` | top differentially expressed genes per cluster (Wilcoxon) |
| `uns[R + "_labels"]` | cluster -> label table with scores; edit it and apply with `ct.tl.apply_manual_labels(adata, R, {"3": "fibroblast"})` (cluster id -> label) |
| `uns["cluster_params"]` | parameters |

## Run metadata: `uns["celltyping"]`

| key | content |
|-----|---------|
| `tissue` | `{"id": "UBERON:0000992", "label": "ovary"}` |
| `species`, `method`, `key`, `label_column` | what was run and where the labels are |
| `tree` | the knowledge tree as JSON; `ct.tl.get_tree(adata)` returns the `CellTypeTree` |
| `n_types_called`, `n_low_quality`, `min_counts`, `runtime_sec` | run summary |

`uns["celltyping"]`, `uns["qc"]` and the `*_params` dicts are written to `.h5ad` without problems.

## Reading the result

```python
adata.obs["hier_label"].value_counts()                     # composition
adata.obs.groupby("hier_level1")["hier_label"].value_counts()   # per lineage
adata.obs.query("hier_label == 'epithelial cell'")         # cells the hierarchy could not resolve further
adata.obs["hier_margin"].describe()                        # confidence distribution
adata.obsm["hier_scores"].groupby(adata.obs["hier_label"]).mean()   # what ct.pl.scores plots
good = adata[adata.obs["qc_pass"]]                         # analyses on high-quality cells only
```

A cell type never appearing in `hier_label` usually means one of: it is not in the tree (check
`ct.tl.get_tree(adata).render()` and the coverage table), it has too few markers on the panel (`*_dropped.csv`), or
its cells stop at the parent because the sibling signatures overlap (lower `min_margin`, or `subtree_mode="max"`).
