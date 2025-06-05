"""Metrics package for evaluating cell perturbation predictions."""

from ._anndata import (
    discrimination_score,
    mae,
    mae_delta,
    mse,
    mse_delta,
    pearson_delta,
)
from ._de import (
    DEDirectionMatch,
    DENsigCounts,
    DEOverlapMetric,
    DESigGenesRecall,
    DESpearmanLFC,
    DESpearmanSignificant,
    PrecisionAt50,
    PrecisionAt100,
    PrecisionAt200,
    Top50Overlap,
    Top100Overlap,
    Top200Overlap,
    TopNOverlap,
    compute_pr_auc,
    compute_roc_auc,
)
from ._impl import metrics_registry
from .base import Metric, MetricInfo, MetricResult

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
    # Global registry
    "metrics_registry",
    # Base Classes
    "Metric",
    "MetricResult",
    "MetricInfo",
]
