"""Gene-panel readers (Xenium ``gene_panel.json``, CosMx/any text list, AnnData)."""

from __future__ import annotations

import json
from pathlib import Path

from ..utils import norm_gene


def load_panel(source) -> list[str]:
    """Return the list of gene symbols in a panel.

    Accepts: a list of genes, a Xenium ``gene_panel.json``, a CSV/TSV/TXT with one gene per
    line or a ``gene``/``symbol``/``name`` column, an ``.h5ad`` file, or an AnnData object.
    """
    if isinstance(source, (list, tuple, set)):
        return sorted({norm_gene(g) for g in source})
    if hasattr(source, "var_names"):
        return sorted({norm_gene(g) for g in source.var_names})
    p = Path(source)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        targets = d.get("payload", {}).get("targets", [])
        genes = []
        for t in targets:
            typ = t.get("type", {})
            if typ.get("descriptor", "gene") != "gene":  # skip negative/genomic controls, deprecated codewords
                continue
            name = typ.get("data", {}).get("name")
            if name and not str(name).lower().startswith(("negcontrol", "unassigned", "blank", "deprecated")):
                genes.append(name)
        return sorted({norm_gene(g) for g in genes})
    if p.suffix == ".h5ad":
        import anndata as ad

        a = ad.read_h5ad(p, backed="r")
        return sorted({norm_gene(g) for g in a.var_names})
    import pandas as pd

    try:
        df = pd.read_csv(p, sep=None, engine="python")
    except Exception:
        df = None
    if df is not None and df.shape[1] > 1:
        col = next((c for c in df.columns if str(c).lower() in ("gene", "symbol", "name", "gene_name", "feature_name")), df.columns[0])
        return sorted({norm_gene(g) for g in df[col].dropna()})
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if lines and lines[0].lower() in ("gene", "symbol", "name"):
        lines = lines[1:]
    return sorted({norm_gene(g) for g in lines})
