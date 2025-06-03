"""CLI module for metric evaluation."""

import logging
from typing import Dict, Set

import anndata as ad
import numpy as np
from pdex import parallel_differential_expression

from ..metrics.pipeline import MetricPipeline
from ..metrics.registry import MetricType, registry
from ..metrics.types import DEComparison, DEResults, PerturbationAnndataPair

logger = logging.getLogger(__name__)


def parse_args_metrics(parser):
    """Parse arguments for the metrics subcommand."""
    parser.add_argument(
        "--real",
        type=str,
        required=True,
        help="Path to real data (AnnData file)",
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to predicted data (AnnData file)",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        required=True,
        help="Column name containing perturbation information",
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        required=True,
        help="Name of control perturbation",
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        help="Column name containing cell type information",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="List of metrics to compute. If not provided, all available metrics will be used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save results (CSV file)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        help="Number of threads to use for parallel processing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="wilcoxon",
        help="Statistical test to use for DE analysis",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
        help="FDR threshold for significant genes",
    )
    parser.add_argument(
        "--minimal-eval",
        action="store_true",
        help="Run minimal evaluation (skip some metrics)",
    )


def _validate_inputs(
    real: ad.AnnData, pred: ad.AnnData, pert_col: str, celltype_col: str
) -> None:
    """Validate input data."""
    # Check gene ordering
    if not np.all(real.var_names.values == pred.var_names.values):
        raise ValueError("Gene ordering is not the same between real and pred data")

    # Check required columns
    for col in [pert_col, celltype_col]:
        if col not in real.obs.columns:
            raise ValueError(
                f"Column '{col}' not found in real data: {real.obs.columns}"
            )
        if col not in pred.obs.columns:
            raise ValueError(
                f"Column '{col}' not found in pred data: {pred.obs.columns}"
            )


def _split_by_celltype(adata: ad.AnnData, celltype_col: str) -> Dict[str, ad.AnnData]:
    """Split an AnnData object by cell type."""
    return {
        ct: adata[adata.obs[celltype_col] == ct]
        for ct in adata.obs[celltype_col].unique()
    }


def _get_celltype_perts(
    adata: ad.AnnData, pert_col: str, celltype_col: str
) -> Dict[str, Set[str]]:
    """Get perturbations per cell type."""
    return adata.obs.groupby(celltype_col)[pert_col].agg(set).to_dict()


def _get_samples(
    adata: ad.AnnData, celltype: str, pert: str, pert_col: str, celltype_col: str
) -> ad.AnnData:
    """Get samples for a specific cell type and perturbation."""
    mask = (adata.obs[celltype_col] == celltype) & (adata.obs[pert_col] == pert)
    return adata[mask]


def _group_indices(
    adata: ad.AnnData, celltype: str, pert_col: str, celltype_col: str
) -> Dict[str, np.ndarray]:
    """Get indices for each perturbation in a cell type."""
    mask = adata.obs[celltype_col] == celltype
    return adata.obs[mask].groupby(pert_col).indices


def run_metrics(args):
    """Run metrics evaluation."""
    # Load data
    logger.info("Loading data...")
    real = ad.read_h5ad(args.real)
    pred = ad.read_h5ad(args.pred)

    # Validate inputs
    _validate_inputs(real, pred, args.pert_col, args.celltype_col)

    # Get perturbations per cell type
    real_celltype_perts = _get_celltype_perts(real, args.pert_col, args.celltype_col)
    pred_celltype_perts = _get_celltype_perts(pred, args.pert_col, args.celltype_col)

    # Ensure matching celltypes and perturbation sets
    if set(real_celltype_perts) != set(pred_celltype_perts):
        raise ValueError("Real and pred data do not share identical celltypes")
    for ct in real_celltype_perts:
        if real_celltype_perts[ct] != pred_celltype_perts[ct]:
            raise ValueError(f"Different perturbations for celltype: {ct}")

    # Initialize pipeline
    pipeline = MetricPipeline()

    # Add metrics
    if args.metrics:
        pipeline.add_metrics(args.metrics)
    else:
        # Add all available metrics
        for metric_type in MetricType:
            pipeline.add_metrics(registry.list_metrics(metric_type))

    real_celltypes = _split_by_celltype(real, args.celltype_col)
    pred_celltypes = _split_by_celltype(pred, args.celltype_col)

    for celltype in real_celltypes:
        ct_real = real_celltypes[celltype]
        ct_pred = pred_celltypes[celltype]

        pair = PerturbationAnndataPair(
            real=ct_real,
            pred=ct_pred,
            pert_col=args.pert_col,
            control_pert=args.control_pert,
        )

        for delta in pair.iter_delta_arrays():
            pipeline.compute_delta_metrics(
                delta,
                celltype=celltype,
            )

        de_real = parallel_differential_expression(
            adata=ct_real,
            reference=args.control_pert,
            groupby_key=args.pert_col,
            metric=args.metric,
            num_workers=args.n_threads,
            batch_size=args.batch_size,
        )
        de_pred = parallel_differential_expression(
            adata=ct_pred,
            reference=args.control_pert,
            groupby_key=args.pert_col,
            metric=args.metric,
            num_workers=args.n_threads,
            batch_size=args.batch_size,
        )

        de_comparison = DEComparison(
            real=DEResults(
                data=de_real,
                control_pert=args.control_pert,
            ),
            pred=DEResults(
                data=de_pred,
                control_pert=args.control_pert,
            ),
        )

        pipeline.compute_de_metrics(
            data=de_comparison,
            celltype=celltype,
        )

    results = pipeline.get_results()
    print(results)

    # # Process each cell type
    # for celltype in tqdm(real_celltype_perts, desc="Processing cell types"):
    #     # Get control samples
    #     real_ctrl = _get_samples(real, celltype, args.control_pert, args.pert_col, args.celltype_col)
    #     pred_ctrl = _get_samples(pred, celltype, args.control_pert, args.pert_col, args.celltype_col)

    #     # Get perturbation groups
    #     real_groups = _group_indices(real, celltype, args.pert_col, args.celltype_col)
    #     pred_groups = _group_indices(pred, celltype, args.pert_col, args.celltype_col)

    #     # Process each perturbation
    #     for pert in tqdm(real_celltype_perts[celltype], desc=f"Processing perturbations for {celltype}", leave=False):
    #         if pert == args.control_pert:
    #             continue

    #         # Get samples
    #         real_idx = real_groups.get(pert, np.array([]))
    #         pred_idx = pred_groups.get(pert, np.array([]))
    #         if real_idx.size == 0 or pred_idx.size == 0:
    #             continue

    #         # Extract arrays
    #         real_pert = to_dense(real[real_idx].X)
    #         pred_pert = to_dense(pred[pred_idx].X)
    #         real_ctrl_arr = to_dense(real_ctrl.X)
    #         pred_ctrl_arr = to_dense(pred_ctrl.X)

    #         # Compute delta metrics
    #         pipeline.compute_delta_metrics(
    #             pert_real=real_pert,
    #             pert_pred=pred_pert,
    #             ctrl_real=real_ctrl_arr,
    #             ctrl_pred=pred_ctrl_arr,
    #             celltype=celltype,
    #             perturbation=pert,
    #         )

    #     # Compute DE metrics if not minimal evaluation
    #     if not args.minimal_eval:
    #         # Subset by celltype
    #         real_ct = real[real.obs[args.celltype_col] == celltype]
    #         pred_ct = pred[pred.obs[args.celltype_col] == celltype]

    #         # Perform DE
    #         de_results = compute_DE_for_truth_and_pred(
    #             real_ct,
    #             pred_ct,
    #             control_pert=args.control_pert,
    #             pert_col=args.pert_col,
    #             celltype_col=args.celltype_col,
    #             n_top_genes=2000,
    #             n_threads=args.n_threads or mp.cpu_count(),
    #             batch_size=args.batch_size,
    #             metric=args.metric,
    #         )

    #         # Create DEResults objects
    #         real_de = DEResults(
    #             data=de_results["real_df"],
    #             control_pert=args.control_pert,
    #         )
    #         pred_de = DEResults(
    #             data=de_results["pred_df"],
    #             control_pert=args.control_pert,
    #         )

    #         # Compute DE metrics
    #         pipeline.compute_de_metrics(
    #             real=real_de,
    #             pred=pred_de,
    #             perturbations=list(real_celltype_perts[celltype]),
    #             celltype=celltype,
    #         )

    # # Get results
    # results = pipeline.get_results()
    # summary = pipeline.get_summary_stats()

    # # Save results
    # logger.info(f"Saving results to {args.output}")
    # output_path = Path(args.output)
    # results.to_csv(output_path)
    # summary.to_csv(output_path.with_suffix(".summary.csv"))
