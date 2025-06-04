"""DE metrics module."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_curve

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
    description="Spearman correlation on number of significant DE genes",
)
class DESpearmanSignificant:
    """Compute Spearman correlation on number of significant DE genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> float:
        """Compute correlation between number of significant genes in real and predicted DE."""

        filt_real = (
            data.real.filter_to_significant(fdr_threshold=self.fdr_threshold)
            .group_by(data.real.target_col)
            .len()
        )
        filt_pred = (
            data.pred.filter_to_significant(fdr_threshold=self.fdr_threshold)
            .group_by(data.pred.target_col)
            .len()
        )

        merged = filt_real.join(
            filt_pred,
            left_on=data.real.target_col,
            right_on=data.pred.target_col,
            suffix="_pred",
            how="full",
            coalesce=True,
        ).fill_null(0)

        # No significant genes in either real or predicted DE. Set to 1.0 since perfect
        # agreement but will fail spearman test
        if merged.shape[0] == 0:
            return 1.0

        return float(
            merged.select(
                pl.corr(
                    pl.col("len"),
                    pl.col("len_pred"),
                    method="spearman",
                ).alias("spearman_corr_nsig")
            )
            .to_numpy()
            .flatten()[0]
        )


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

        merged = data.real.filter_to_significant(fdr_threshold=0.05).join(
            data.pred.data,
            on=[data.real.target_col, data.real.feature_col],
            suffix="_pred",
            how="inner",
        )
        for row in (
            merged.with_columns(
                direction_match=pl.col(data.real.log2_fold_change_col).sign()
                == pl.col(f"{data.real.log2_fold_change_col}_pred").sign()
            )
            .group_by(
                data.real.target_col,
            )
            .agg(pl.mean("direction_match"))
            .iter_rows()
        ):
            matches.update({row[0]: row[1]})
        return matches


@registry.register(
    name="de_spearman_lfc_sig",
    metric_type=MetricType.DE,
    description="Spearman correlation on log fold changes of significant genes",
)
class DESpearmanLFC:
    """Compute Spearman correlation on log fold changes of significant genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> Dict[str, float]:
        """Compute correlation between log fold changes of significant genes."""
        correlations = {}

        merged = data.real.filter_to_significant(fdr_threshold=self.fdr_threshold).join(
            data.pred.data,
            on=[data.real.target_col, data.real.feature_col],
            suffix="_pred",
            how="inner",
        )

        for row in (
            merged.group_by(
                data.real.target_col,
            )
            .agg(
                pl.corr(
                    pl.col(data.real.fold_change_col),
                    pl.col(f"{data.real.fold_change_col}_pred"),
                    method="spearman",
                ).alias("spearman_corr"),
            )
            .iter_rows()
        ):
            correlations.update({row[0]: row[1]})

        return correlations


@registry.register(
    name="de_pr_auc",
    metric_type=MetricType.DE,
    description="Precision-Recall AUC for significant genes",
)
class DEPRAUC:
    """Compute Precision-Recall AUC for significant genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> Dict[str, float]:
        """Compute PR AUC between real and predicted significant genes."""
        pr_aucs = {}

        for pert in data.iter_perturbations():
            real_sig = data.real.filter_to_significant(fdr_threshold=self.fdr_threshold)
            pred_sig = data.pred.filter_to_significant(fdr_threshold=self.fdr_threshold)

            merged = real_sig.merge(
                pred_sig,
                on=[data.real.target_col, data.real.feature_col],
                suffixes=("_real", "_pred"),
                how="outer",
            )

            if merged.shape[0] == 0:
                pr_aucs[pert] = np.nan
                continue

            # Create binary labels based on real significance
            y = (merged[f"{data.real.fdr_col}_real"] < self.fdr_threshold).astype(int)
            scores = -np.log10(merged[f"{data.pred.fdr_col}_pred"])

            if 0 < y.sum() < len(y):
                pr, re, _ = precision_recall_curve(y, scores)
                pr_aucs[pert] = float(auc(re, pr))
            else:
                pr_aucs[pert] = np.nan

        return pr_aucs


@registry.register(
    name="de_roc_auc",
    metric_type=MetricType.DE,
    description="ROC AUC for significant genes",
)
class DEROCAUC:
    """Compute ROC AUC for significant genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> Dict[str, float]:
        """Compute ROC AUC between real and predicted significant genes."""
        roc_aucs = {}

        for pert in data.iter_perturbations():
            real_sig = data.real.filter_to_significant(fdr_threshold=self.fdr_threshold)
            pred_sig = data.pred.filter_to_significant(fdr_threshold=self.fdr_threshold)

            merged = real_sig.merge(
                pred_sig,
                on=[data.real.target_col, data.real.feature_col],
                suffixes=("_real", "_pred"),
                how="outer",
            )

            if merged.shape[0] == 0:
                roc_aucs[pert] = np.nan
                continue

            # Create binary labels based on real significance
            y = (merged[f"{data.real.fdr_col}_real"] < self.fdr_threshold).astype(int)
            scores = -np.log10(merged[f"{data.pred.fdr_col}_pred"])

            if 0 < y.sum() < len(y):
                f, t, _ = roc_curve(y, scores)
                roc_aucs[pert] = float(auc(f, t))
            else:
                roc_aucs[pert] = np.nan

        return roc_aucs


@registry.register(
    name="de_sig_genes_recall",
    metric_type=MetricType.DE,
    description="Recall of significant genes",
)
class DESigGenesRecall:
    """Compute recall of significant genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> Dict[str, float]:
        """Compute recall of significant genes between real and predicted DE."""
        recalls = {}

        for pert in data.iter_perturbations():
            real_sig = data.real.get_significant_genes(pert, self.fdr_threshold)
            pred_sig = data.pred.get_significant_genes(pert, self.fdr_threshold)

            if real_sig.size == 0:
                recalls[pert] = 0.0
                continue

            recalls[pert] = float(
                np.intersect1d(real_sig, pred_sig).size / real_sig.size
            )

        return recalls


@registry.register(
    name="de_nsig_counts",
    metric_type=MetricType.DE,
    description="Counts of significant genes",
)
class DENsigCounts:
    """Compute counts of significant genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> Dict[str, Dict[str, int]]:
        """Compute counts of significant genes in real and predicted DE."""
        counts = {}

        for pert in data.iter_perturbations():
            real_sig = data.real.get_significant_genes(pert, self.fdr_threshold)
            pred_sig = data.pred.get_significant_genes(pert, self.fdr_threshold)

            counts[pert] = {
                "real": int(real_sig.size),
                "pred": int(pred_sig.size),
            }

        return counts
