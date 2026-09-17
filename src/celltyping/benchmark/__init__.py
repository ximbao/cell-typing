"""Benchmark: compare annotation strategies against an expert reference mapped onto the knowledge tree."""

from .harmonize import project_to_reference_classes, reference_ids, relation_table  # noqa: F401
from .metrics import agreement, hierarchical_prf, marker_specificity, spatial_coherence  # noqa: F401
from .run import run_benchmark  # noqa: F401
