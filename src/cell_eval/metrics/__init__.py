"""Metrics package for evaluating cell perturbation predictions."""

from .array_metrics import (
    # MMD,
    # Wasserstein,
    # mse,
    # pearson_correlation,
    pearson_delta,
)
from .de_metrics import (
    DEPRAUC,
    DEROCAUC,
    DEDirectionMatch,
    DENsigCounts,
    DEOverlapMetric,
    DESigGenesRecall,
    DESpearmanLFC,
    DESpearmanSignificant,
    SignificantGeneOverlap,
    Top50Overlap,
    Top100Overlap,
    Top200Overlap,
)
from .registry import MetricRegistry, MetricType, registry
from .types import (
    DEComparison,
    DeltaArrays,
    DEResults,
    DESortBy,
    PerturbationAnndataPair,
)

__all__ = [
    # Array metrics
    # "MMD",
    # "Wasserstein",
    # "mse",
    # "pearson_correlation",
    "pearson_delta",
    # DE metrics
    "DEDirectionMatch",
    "DEOverlapMetric",
    "DESpearmanSignificant",
    "SignificantGeneOverlap",
    "Top50Overlap",
    "Top100Overlap",
    "Top200Overlap",
    "DESpearmanLFC",
    "DEPRAUC",
    "DEROCAUC",
    "DESigGenesRecall",
    "DENsigCounts",
    # Registry
    "MetricRegistry",
    "MetricType",
    "registry",
    # Types
    "DEComparison",
    "DEResults",
    "DESortBy",
    "DeltaArrays",
    "PerturbationAnndataPair",
]
