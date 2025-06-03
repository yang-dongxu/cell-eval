"""DE metrics module."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .registry import MetricType, registry
from .types import DEComparison, DESortBy


@dataclass
class DEOverlapMetric:
    """Base class for overlap-based DE metrics."""

    k: Optional[int] = None
    topk: Optional[int] = None
    fdr_threshold: Optional[float] = 0.05
    sort_by: DESortBy = DESortBy.ABS_FOLD_CHANGE

    def __call__(self, data: DEComparison) -> Dict[str, float]:
        """Compute overlap between real and predicted DE genes."""
        return data.compute_overlap(
            k=self.k,
            topk=self.topk,
            fdr_threshold=self.fdr_threshold,
            sort_by=self.sort_by,
        )


@registry.register(
    name="top_50_overlap",
    metric_type=MetricType.DE,
    description="Overlap of top 50 DE genes",
)
class Top50Overlap(DEOverlapMetric):
    """Compute overlap of top 50 DE genes."""

    def __init__(self) -> None:
        super().__init__(k=50)


@registry.register(
    name="top_100_overlap",
    metric_type=MetricType.DE,
    description="Overlap of top 100 DE genes",
)
class Top100Overlap(DEOverlapMetric):
    """Compute overlap of top 100 DE genes."""

    def __init__(self) -> None:
        super().__init__(k=100)


@registry.register(
    name="top_200_overlap",
    metric_type=MetricType.DE,
    description="Overlap of top 200 DE genes",
)
class Top200Overlap(DEOverlapMetric):
    """Compute overlap of top 200 DE genes."""

    def __init__(self) -> None:
        super().__init__(k=200)


@registry.register(
    name="significant_gene_overlap",
    metric_type=MetricType.DE,
    description="Overlap of all significant DE genes",
)
class SignificantGeneOverlap(DEOverlapMetric):
    """Compute overlap of all significant DE genes."""

    def __init__(self) -> None:
        super().__init__(k=-1)  # -1 means use all significant genes


@registry.register(
    name="de_spearman_sig",
    metric_type=MetricType.DE,
    description="Spearman correlation of significant DE gene counts",
)
class DESpearmanSignificant:
    """Compute Spearman correlation of significant DE gene counts."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> float:
        """Compute correlation between real and predicted significant gene counts."""
        real_counts = {}
        pred_counts = {}

        for pert in data.perturbations:
            if pert == data.real.control_pert:
                continue

            real_sig = data.real.get_significant_genes(pert, self.fdr_threshold)
            pred_sig = data.pred.get_significant_genes(pert, self.fdr_threshold)

            real_counts[pert] = len(real_sig)
            pred_counts[pert] = len(pred_sig)

        if not real_counts:  # No perturbations found
            return 0.0

        real_values = [real_counts[p] for p in real_counts.keys()]
        pred_values = [pred_counts[p] for p in real_counts.keys()]

        return float(spearmanr(real_values, pred_values)[0])


@registry.register(
    name="de_direction_match",
    metric_type=MetricType.DE,
    description="Agreement in direction of DE gene changes",
)
class DEDirectionMatch:
    """Compute agreement in direction of DE gene changes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> Dict[str, float]:
        """Compute directional agreement between real and predicted DE genes."""
        matches = {}

        for pert in data.perturbations:
            if pert == data.real.control_pert:
                continue

            # Get significant genes from real data
            real_data = data.real.data[
                (data.real.data[data.real.target_col] == pert)
                & (data.real.data[data.real.fdr_col] < self.fdr_threshold)
            ]

            # Get corresponding predictions
            real_features = set(real_data[data.real.feature_col])
            pred_data = data.pred.data[
                (data.pred.data[data.pred.target_col] == pert)
                & (data.pred.data[data.pred.feature_col].isin(real_features))
            ]

            # Merge and compare directions
            merged = pd.merge(
                real_data,
                pred_data,
                on=[data.real.target_col, data.real.feature_col],
                suffixes=("_real", "_pred"),
            )

            if len(merged) == 0:
                matches[pert] = 0.0
                continue

            # Compare signs of fold changes
            real_signs = np.sign(merged[f"{data.real.fold_change_col}_real"])
            pred_signs = np.sign(merged[f"{data.pred.fold_change_col}_pred"])
            matches[pert] = float(np.mean(real_signs == pred_signs))

        return matches
