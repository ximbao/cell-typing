"""celltyping: ontology-driven hierarchical cell typing for imaging-based spatial transcriptomics.

Scanpy-style API::

    import celltyping as ct

    adata = ct.read.xenium("outs/")                 # or ct.read.cosmx / ct.read.h5ad
    adata = ct.annotate(adata, tissue="ovary")      # preprocess (if needed) + knowledge + hierarchical annotation
    ct.pl.spatial(adata, "hier_label")

    # or step by step
    adata = ct.pp.preprocess(adata)
    tree = ct.tl.knowledge("ovary", panel=adata)
    ct.tl.hierarchical(adata, tree); ct.tl.flat(adata, tree); ct.tl.clusters(adata, tree)
    ct.pl.tree(tree); ct.pl.composition(adata); ct.pl.scores(adata, tree=tree)
"""

from importlib.metadata import PackageNotFoundError, version

from . import bm, pl, pp, read, tl  # noqa: F401
from ._annotate import annotate  # noqa: F401
from .knowledge.tree import CellTypeTree  # noqa: F401
from .settings import settings  # noqa: F401

try:
    __version__ = version("celltyping")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__", "annotate", "settings", "read", "pp", "tl", "pl", "bm", "CellTypeTree"]
