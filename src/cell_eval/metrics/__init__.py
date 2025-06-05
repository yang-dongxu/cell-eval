"""Metrics package for evaluating cell perturbation predictions."""

from .anndata_metrics import (
    discrimination_score,
    mae,
    mae_delta,
    mse,
    mse_delta,
    pearson_delta,
)
from .de_metrics import (
    DEDirectionMatch,
    DENsigCounts,
    DEOverlapMetric,
    DESigGenesRecall,
    DESpearmanLFC,
    DESpearmanSignificant,
    PrecisionAt50,
    PrecisionAt100,
    PrecisionAt200,
    SignificantGeneOverlap,
    Top50Overlap,
    Top100Overlap,
    Top200Overlap,
    TopNOverlap,
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
    "pearson_delta",
    "mse",
    "mae",
    "mse_delta",
    "mae_delta",
    "discrimination_score",
    # DE metrics
    "DEDirectionMatch",
    "DEOverlapMetric",
    "DESpearmanSignificant",
    "SignificantGeneOverlap",
    "Top50Overlap",
    "Top100Overlap",
    "Top200Overlap",
    "DESpearmanLFC",
    "TopNOverlap",
    "PrecisionAt50",
    "PrecisionAt100",
    "PrecisionAt200",
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
