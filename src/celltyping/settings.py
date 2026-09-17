"""Global settings, scanpy-style: ``celltyping.settings.data_dir = ...``."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .utils import log

_PKG_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    """Package-wide configuration.

    Attributes
    ----------
    data_dir
        Cache for downloaded ontologies, marker databases and Census/CellGuide tables
        (default ``$CELLTYPING_DATA`` or ``<repo>/data``).
    knowledge_dir
        Where :func:`celltyping.tl.knowledge` caches built trees (default ``data_dir / "knowledge"``).
    verbosity
        0 = warnings only, 1 = info (default), 2 = debug.
    species
        Default organism (``'human'`` or ``'mouse'``).
    census / census_version
        Whether :func:`celltyping.tl.knowledge` uses CZ CELLxGENE Census observations as tissue
        evidence, and which Census release.
    sources
        Default marker sources.
    """

    def __init__(self) -> None:
        self._data_dir: Path | None = None
        self._knowledge_dir: Path | None = None
        self._verbosity = 1
        self.species = "human"
        self.census = True
        self.census_version: str | None = None  # None -> pinned default in knowledge.cellxgene
        self.sources: tuple[str, ...] = ("CellMarker2", "PanglaoDB", "ASCT+B", "CellGuide")
        self.random_state = 0

    # -- paths
    @property
    def data_dir(self) -> Path:
        d = self._data_dir or Path(os.environ.get("CELLTYPING_DATA", _PKG_ROOT / "data"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    @data_dir.setter
    def data_dir(self, value: str | Path) -> None:
        self._data_dir = Path(value)

    @property
    def knowledge_dir(self) -> Path:
        d = self._knowledge_dir or self.data_dir / "knowledge"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @knowledge_dir.setter
    def knowledge_dir(self, value: str | Path) -> None:
        self._knowledge_dir = Path(value)

    # -- logging
    @property
    def verbosity(self) -> int:
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value: int) -> None:
        self._verbosity = int(value)
        log.setLevel({0: logging.WARNING, 1: logging.INFO}.get(self._verbosity, logging.DEBUG))

    def __repr__(self) -> str:
        return (f"celltyping.settings(data_dir={str(self.data_dir)!r}, knowledge_dir={str(self.knowledge_dir)!r}, "
                f"verbosity={self.verbosity}, species={self.species!r}, census={self.census}, sources={self.sources})")


settings = Settings()
