"""``celltyping.read``: loaders for imaging-based spatial transcriptomics outputs.

All readers return an :class:`anndata.AnnData` with raw counts in ``X``, spatial coordinates in
``obsm['spatial']`` and control probes (negative controls, blanks, ...) moved to ``obs['control_counts']``.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from .io import read_cosmx as cosmx  # noqa: F401
from .io import read_dataset as dataset  # noqa: F401
from .io import read_xenium as xenium  # noqa: F401


def h5ad(path: str | Path) -> ad.AnnData:
    """Read an ``.h5ad`` file (thin wrapper around :func:`anndata.read_h5ad`)."""
    return ad.read_h5ad(path)


__all__ = ["xenium", "cosmx", "dataset", "h5ad"]
