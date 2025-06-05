from .._types import MetricType
from ._anndata import (
    ClusteringAgreement,
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
from ._registry import MetricRegistry

metrics_registry = MetricRegistry()

metrics_registry.register(
    name="pearson_delta",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Pearson correlation between mean differences from control",
    func=pearson_delta,
)

metrics_registry.register(
    name="mse",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean squared error of each perturbation from control.",
    func=mse,
)
metrics_registry.register(
    name="mae",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean absolute error of each perturbation from control.",
    func=mae,
)

metrics_registry.register(
    name="mse_delta",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean squared error of each perturbation-control delta.",
    func=mse_delta,
)

metrics_registry.register(
    name="mae_delta",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean squared error of each perturbation-control delta.",
    func=mae_delta,
)

metrics_registry.register(
    name="discrimination_score",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Determines similarity of each pred representation to real via normalized rank of cosine similarity",
    func=discrimination_score,
)
metrics_registry.register(
    name="top_N_overlap",
    metric_type=MetricType.DE,
    description="Overlap of top k DE genes",
    func=TopNOverlap,
    is_class=True,
)

metrics_registry.register(
    name="top_50_overlap",
    metric_type=MetricType.DE,
    description="Overlap of top 50 DE genes",
    func=Top50Overlap,
    is_class=True,
)

metrics_registry.register(
    name="top_100_overlap",
    metric_type=MetricType.DE,
    description="Overlap of top 100 DE genes",
    func=Top100Overlap,
    is_class=True,
)

metrics_registry.register(
    name="top_200_overlap",
    metric_type=MetricType.DE,
    description="Overlap of top 200 DE genes",
    func=Top200Overlap,
    is_class=True,
)

metrics_registry.register(
    name="precision_at_50",
    metric_type=MetricType.DE,
    description="Precision at 50",
    func=PrecisionAt50,
    is_class=True,
)

metrics_registry.register(
    name="precision_at_100",
    metric_type=MetricType.DE,
    description="Precision at 100",
    func=PrecisionAt100,
    is_class=True,
)

metrics_registry.register(
    name="precision_at_200",
    metric_type=MetricType.DE,
    description="Precision at 200",
    func=PrecisionAt200,
    is_class=True,
)

metrics_registry.register(
    name="de_spearman_sig",
    metric_type=MetricType.DE,
    description="Spearman correlation on number of significant DE genes",
    func=DESpearmanSignificant,
    is_class=True,
)

metrics_registry.register(
    name="de_direction_match",
    metric_type=MetricType.DE,
    description="Agreement in direction of DE gene changes",
    func=DEDirectionMatch,
    is_class=True,
)

metrics_registry.register(
    name="de_spearman_lfc_sig",
    metric_type=MetricType.DE,
    description="Spearman correlation on log fold changes of significant genes",
    func=DESpearmanLFC,
    is_class=True,
)

metrics_registry.register(
    name="de_sig_genes_recall",
    metric_type=MetricType.DE,
    description="Recall of significant genes",
    func=DESigGenesRecall,
    is_class=True,
)

metrics_registry.register(
    name="de_nsig_counts",
    metric_type=MetricType.DE,
    description="Counts of significant genes",
    func=DENsigCounts,
    is_class=True,
)

metrics_registry.register(
    name="pr_auc",
    metric_type=MetricType.DE,
    description="Computes precision-recall for significant recovery",
    func=compute_pr_auc,
)

metrics_registry.register(
    name="roc_auc",
    metric_type=MetricType.DE,
    description="Computes ROC AUC for significant recovery",
    func=compute_roc_auc,
)

metrics_registry.register(
    name="clustering_agreement",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Clustering agreement between real and predicted perturbation centroids",
    func=ClusteringAgreement,
    is_class=True,
)
