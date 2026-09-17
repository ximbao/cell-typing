"""Shared helpers: paths, downloads, logging."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

log = logging.getLogger("celltyping")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(os.environ.get("CELLTYPING_LOGLEVEL", "INFO"))

_PKG_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Root cache directory for downloaded ontologies and marker databases.

    Set via ``celltyping.settings.data_dir`` or the ``CELLTYPING_DATA`` environment variable.
    """
    from .settings import settings  # local import: settings imports utils.log

    return settings.data_dir


def download(url: str, dest: Path, force: bool = False, timeout: int = 300) -> Path:
    """Download ``url`` to ``dest`` unless it already exists. Returns ``dest``."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", url, dest)
    headers = {"User-Agent": "Mozilla/5.0 (celltyping)"}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    return dest


def norm_gene(symbol: str) -> str:
    """Normalise a gene symbol for case-insensitive matching (human/mouse panels)."""
    return str(symbol).strip().upper()


def norm_label(label: str) -> str:
    """Normalise a cell-type label for fuzzy string matching."""
    s = str(label).strip().lower()
    for ch in ("-", "_", "/", ",", "."):
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    if s.endswith("cells"):
        s = s[:-1]
    elif s.endswith("s") and not s.endswith("ss") and not s.endswith("us"):
        s = s[:-1]
    return s
