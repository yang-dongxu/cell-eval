"""Metrics package for evaluating cell perturbation predictions."""

from .array_metrics import (
    MMD,
    Wasserstein,
    mse,
    pearson_correlation,
    pearson_delta,
)
from .de_metrics import (
    DEDirectionMatch,
    DEOverlapMetric,
    DESpearmanSignificant,
    SignificantGeneOverlap,
    Top50Overlap,
    Top100Overlap,
    Top200Overlap,
)
from .registry import MetricRegistry, MetricType, registry
from .types import (
    AnnDataPair,
    ArrayPair,
    DEComparison,
    DeltaArrays,
    DEResults,
    DESortBy,
    PerturbationAnnData,
)

__all__ = [
    # Array metrics
    "MMD",
    "Wasserstein",
    "mse",
    "pearson_correlation",
    "pearson_delta",
    # DE metrics
    "DEDirectionMatch",
    "DEOverlapMetric",
    "DESpearmanSignificant",
    "SignificantGeneOverlap",
    "Top50Overlap",
    "Top100Overlap",
    "Top200Overlap",
    # Registry
    "MetricRegistry",
    "MetricType",
    "registry",
    # Types
    "AnnDataPair",
    "ArrayPair",
    "DEComparison",
    "DEResults",
    "DESortBy",
    "DeltaArrays",
    "PerturbationAnnData",
]
