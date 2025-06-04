"""Metrics package for evaluating cell perturbation predictions."""

from .array_metrics import (
    # MMD,
    # Wasserstein,
    # mse,
    # pearson_correlation,
    pearson_delta,
)
from .de_metrics import (
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
    compute_pr_auc,
    compute_roc_auc,
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
    "compute_pr_auc",
    "compute_roc_auc",
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
