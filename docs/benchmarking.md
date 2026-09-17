# Benchmarking against a dataset with known cell types

When a dataset carries reference labels (expert annotation, a validated pipeline, matched scRNA-seq transfer),
`celltyping` can score its own annotations -- and the flat and clustering baselines -- against them. The
example is the Xenium Prime 5K ovarian carcinoma section whose `obs["cell_type"]` holds an expert annotation
with 18 labels; the procedure is the same for any dataset and any reference column.

The benchmark evaluates **high-quality cells only** (`obs["qc_pass"]`), and only cells whose reference label
could be mapped to the tree.

## 1. Map the reference labels onto the ontology

Reference labels are free text ("Tumor Associated Fibroblasts", "T and NK Cells"); predictions are Cell Ontology
nodes. The bridge is a **reference map**: label -> CL id. Choose the CL term that best matches the *meaning* of
the label; it need not be in the tree -- ids that are absent are lifted to their nearest ancestor that is
(e.g. `T cell` -> `lymphocyte` if only `lymphocyte` exists), and labels you do not want to evaluate go to
`ignore_labels`.

```python
reference_map = {
    "Tumor Cells":                          "CL:0001064",   # malignant cell
    "Proliferative Tumor Cells":            "CL:0001064",   # several reference labels may map to one node
    "Fallopian Tube Epithelium":            "CL:4052018",   # fallopian tube epithelial cell
    "Ciliated Epithelial Cells":            "CL:4030007",   # fallopian tube multiciliated epithelial cell
    "Urothelial-like Cells":                "CL:0000731",
    "Tumor Associated Fibroblasts":         "CL:0000057",   # fibroblast
    "Stromal Associated Fibroblasts":       "CL:0000057",
    "Smooth Muscle Cells":                  "CL:0000192",
    "Pericytes":                            "CL:0000669",
    "Stromal Associated Endothelial Cells": "CL:0000115",
    "Tumor Associated Endothelial Cells":   "CL:0000115",
    "Macrophages":                          "CL:0000235",
    "T and NK Cells":                       "CL:0000542",   # lymphocyte: the reference does not split T from NK
}
ignore_labels = ["Unassigned"]
```

Finding ids: `celltyping search-tissue` is for tissues; for cell types use `ct.tl.expected_cell_types(tissue)`
(lists the CL ids the ontology and the Census expect) or the [OLS Cell Ontology browser](https://www.ebi.ac.uk/ols4/ontologies/cl).

## 2. Python: `ct.bm`

```python
import celltyping as ct

adata = ct.read.h5ad("ov_validation.h5ad")
ct.annotate(adata, tissue="ovary", overrides="configs/overrides_ovary.yaml")            # obs['hier_*']
ct.annotate(adata, tissue="ovary", overrides="configs/overrides_ovary.yaml", method="flat")     # obs['flat_*']
ct.annotate(adata, tissue="ovary", overrides="configs/overrides_ovary.yaml", method="cluster")  # obs['cluster_r*']
tree = ct.tl.get_tree(adata)

ref = ct.bm.reference(adata, "cell_type", reference_map, tree, ignore=ignore_labels)     # obs['ref_id']
summary = ct.bm.compare(adata, ref, tree, methods=("hier", "flat", "cluster"))
print(summary[["n_eval", "lineage_acc", "exact_acc", "coarser", "unassigned", "wrong", "hF1", "macro_f1",
               "ARI_assigned", "spatial_coherence", "n_labels"]].round(3))
```

```
              n_eval  lineage_acc  exact_acc  coarser  unassigned  wrong    hF1  macro_f1  ARI_assigned  spatial_coherence  n_labels
hier          382294        0.787      0.743    0.022       0.038  0.153  0.787     0.481         0.677              0.565        22
flat          382294        0.806      0.655    0.000       0.025  0.169  0.743     0.528         0.642              0.594        16
cluster_r0.5  382294        0.827      0.694    0.000       0.003  0.170  0.757     0.418         0.719              0.652         7
cluster_r1    382294        0.774      0.680    0.000       0.011  0.216  0.726     0.411         0.692              0.612         9
```

Per-class and confusion views:

```python
summ, per_class, conf = ct.bm.agreement(adata.obs["hier_id"], ref, tree)
per_class[["ref_label", "n_ref", "precision", "recall", "f1", "coarser", "unassigned", "wrong"]]
ct.pl.confusion(ct.bm.confusion(adata, ref, tree, key="hier"), title="hierarchical vs expert")

# side-by-side spatial maps with a shared palette
colors = ct.pl.palette(list(adata.obs["hier_label"]) + list(ct.bm.labels_of(ref, tree)), tree)
ct.pl.spatial(adata, "hier_label", colors=colors)
adata.obs["ref_label"] = ct.bm.labels_of(ref, tree)
ct.pl.spatial(adata, "ref_label", colors=colors)
```

Label-free metrics are available for datasets without a reference, or to judge the reference itself:

```python
ct.bm.spatial_coherence(adata.obsm["spatial"], adata.obs["hier_id"], tree, k=10)
ct.bm.marker_specificity(adata, adata.obs["hier_id"], tree, layer="knn_smooth")
ct.bm.label_entropy(adata.obs["hier_id"], tree)
```

## 3. Command line: `celltyping benchmark`

Put the reference map in the run config and let the CLI annotate, evaluate and write a report:

```yaml
# configs/ovarian_xenium.yaml (relevant parts)
tissue: ovary
panel: /path/to/gene_panel.json
output_dir: ../results/ovarian_xenium

knowledge:
  overrides: overrides_ovary.yaml

datasets:
  - name: ov_validation
    path: /path/to/ov_validation.h5ad
    platform: xenium

preprocess:
  min_counts: 20            # QC flag threshold
  knn_smooth: 15

benchmark:
  reference_column: cell_type
  ignore_labels: [Unassigned]
  reference_map:
    "Tumor Cells": "CL:0001064"
    # ...
  spatial_k: 10
```

```bash
celltyping benchmark --config configs/ovarian_xenium.yaml --dataset ov_validation \
    --methods hier,flat,cluster [--subsample 50000] [--no-reuse] [--force-knowledge]
```

Outputs in `results/ovarian_xenium/ov_validation/`:

| file | content |
|------|---------|
| `annotated.h5ad`, `labels.csv` | the annotated dataset (all methods, QC flag, coordinates, reference column) |
| `benchmark/summary.csv`, `summary.png` | one row per method (plus the reference's own label-free metrics) |
| `benchmark/per_class_<method>.csv` | precision / recall / F1 and error breakdown per reference class |
| `benchmark/confusion_<method>.csv/.png` | reference x prediction matrices |
| `benchmark/spatial_labels.png` | expert map next to each method's map |
| `benchmark/report.md` | everything above as markdown, with the tree and metric definitions |

`--subsample N` evaluates on a random subset (fast iteration on override files); `--no-reuse` re-annotates even
if `annotated.h5ad` exists; methods missing from a saved run are added without redoing the others.

## 4. Reading the metrics

Predictions and references are both tree nodes, so agreement is hierarchical. For each evaluated cell the
relation between prediction *p* and reference *r* is one of:

| relation | meaning | counted in |
|----------|---------|-----------|
| exact | `p == r` | `exact_acc`, `lineage_acc` |
| descendant | *p* is a descendant of *r* (the method is more specific than the reference, e.g. `T cell` vs `lymphocyte`) | `lineage_acc` |
| coarser | *p* is a strict ancestor of *r* (the method stopped early, e.g. `leukocyte` vs `macrophage`) | `coarser` |
| unassigned | root, `unassigned`, or not in the tree | `unassigned` |
| wrong | anything else | `wrong` |

`lineage_acc + coarser + unassigned + wrong = 1`. **`lineage_acc`** is the headline number: the prediction is right
*at the reference's granularity*. `coarser` is not an error in the hierarchical sense but tells you how often the
method refused to commit.

- **hP / hR / hF1**: hierarchical precision, recall and F1 -- the overlap of the ancestor sets of *p* and *r*
  (Kiritchenko et al.), so a `leukocyte` call for a macrophage gets partial credit and a `fibroblast` call gets none.
- **per-class precision / recall / F1**: computed on predictions *projected* onto the reference classes (a `T cell`
  prediction counts as `lymphocyte` when `lymphocyte` is the reference class), so methods with different
  granularity are compared fairly; `macro_f1` is their unweighted mean and rewards finding rare types.
- **ARI / NMI** (`_assigned`: on assigned cells; `_all`: unassigned counted as a class): clustering agreement
  independent of label names.
- **spatial_coherence**: fraction of a cell's `k` nearest spatial neighbours (among assigned cells) sharing its
  label. Higher is smoother, not necessarily better -- compare with the reference's own value (0.63 here).
- **marker_specificity**: how well each type's own markers separate its cells from the rest; label-free.
- **n_labels / label_entropy_bits**: how many types a method actually called and how evenly.

Typical patterns on the ovarian data: clustering scores highest on `lineage_acc` because entire clusters are
labelled at once for the five abundant types, but it never produces rare types (`n_labels` 7); the hierarchical
annotator has the highest exact agreement and hF1; flat leaf scoring finds more rare epithelial and lymphoid
cells but confuses stroma and vessels more. Reading `per_class_*.csv` and the confusion matrices is what turns the
numbers into override-file edits (add a missing type, replace generic markers, force a parent).

## 5. Benchmarking without a full reference

- A partial reference (only some cells labelled, e.g. from pathologist regions) works as is: unlabelled cells are
  excluded from `n_eval`.
- Two methods can be compared against each other: `ct.bm.agreement(adata.obs["flat_id"], adata.obs["hier_id"], tree)`.
- Reference labels from a different vocabulary (e.g. Azimuth or CellTypist names) map the same way; when a label
  does not correspond to a single CL term, map it to the closest common ancestor and the hierarchical metrics
  handle the granularity difference.
