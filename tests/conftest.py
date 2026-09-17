import numpy as np
import pandas as pd
import pytest
import anndata as ad

from celltyping.knowledge.tree import CellTypeTree

# toy hierarchy:
# cell
# ├─ neuron            (SNAP25, RBFOX3, SYT1)
# │  ├─ excitatory     (SLC17A7, SATB2, CUX2)
# │  └─ inhibitory     (GAD1, GAD2, SST)
# └─ glia              (SOX9, SOX10, ALDH1L1)   <- lifted/own
#    ├─ astrocyte      (GFAP, AQP4, ALDH1L1)
#    └─ oligodendrocyte(MBP, PLP1, MOG)
TOY = {
    "neuron": ("N", None, ["SNAP25", "RBFOX3", "SYT1"]),
    "excitatory": ("N_EX", "N", ["SLC17A7", "SATB2", "CUX2"]),
    "inhibitory": ("N_IN", "N", ["GAD1", "GAD2", "SST"]),
    "glia": ("G", None, ["SOX9", "SOX10", "ALDH1L1"]),
    "astrocyte": ("G_AS", "G", ["GFAP", "AQP4", "ALDH1L1"]),
    "oligodendrocyte": ("G_OL", "G", ["MBP", "PLP1", "MOG"]),
}
LEAVES = ["N_EX", "N_IN", "G_AS", "G_OL"]


@pytest.fixture
def toy_tree() -> CellTypeTree:
    t = CellTypeTree()
    for label, (nid, parent, genes) in TOY.items():
        t.add(nid, label, parent, seed=True)
        for g in genes:
            t.add_marker(nid, g, 1.0, "toy")
    return t


@pytest.fixture
def toy_adata(toy_tree) -> ad.AnnData:
    """Synthetic log-normalised data: 4 leaf populations expressing their own + parent markers, plus noise genes."""
    rng = np.random.default_rng(0)
    genes = sorted({g for n in toy_tree.nodes.values() for g in n.markers} | {f"NOISE{i}" for i in range(20)})
    gi = {g: i for i, g in enumerate(genes)}
    n_per = 150
    X = rng.poisson(0.3, size=(n_per * len(LEAVES), len(genes))).astype(float)
    truth = []
    for k, leaf in enumerate(LEAVES):
        rows = slice(k * n_per, (k + 1) * n_per)
        for nid in toy_tree.path(leaf)[1:]:
            for g in toy_tree.nodes[nid].markers:
                X[rows, gi[g]] += rng.poisson(6.0, size=n_per)
        truth += [leaf] * n_per
    a = ad.AnnData(X=np.log1p(X), obs=pd.DataFrame({"truth": truth}, index=[f"c{i}" for i in range(X.shape[0])]), var=pd.DataFrame(index=genes))
    a.obsm["spatial"] = np.column_stack([np.repeat(np.arange(len(LEAVES)) * 100.0, n_per) + rng.normal(0, 10, len(truth)), rng.normal(0, 10, len(truth))])
    return a
