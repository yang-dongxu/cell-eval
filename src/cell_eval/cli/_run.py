import argparse as ap
import logging
import os

logger = logging.getLogger(__name__)


def parse_args_run(parser: ap.ArgumentParser):
    """
    CLI for evaluation
    """
    parser.add_argument(
        "-ap",
        "--adata-pred",
        type=str,
        help="Path to the predicted adata object to evaluate",
        required=True,
    )
    parser.add_argument(
        "-ar",
        "--adata-real",
        type=str,
        help="Path to the real adata object to evaluate against",
        required=True,
    )
    parser.add_argument(
        "-dp",
        "--de-pred",
        type=str,
        help="Path to the predicted DE results (computed with pdex from adata-pred if not provided)",
        required=False,
    )
    parser.add_argument(
        "-dr",
        "--de-real",
        type=str,
        help="Path to the real DE results (computed with pdex from adata-real if not provided)",
        required=False,
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default="non-targeting",
        help="Name of the control perturbation",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default="target_name",
        help="Name of the column designated perturbations",
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        help="Name of the column designated celltype (optional)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="./cell-eval-outdir",
        help="Output directory to write to",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--skip-normlog-check",
        action="store_true",
    )
    parser.add_argument(
        "--de-method",
        type=str,
        default="wilcoxon",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
    )


def build_outdir(outdir: str):
    if os.path.exists(outdir):
        logger.warning(
            f"Output directory {outdir} already exists, potential overwrite occurring"
        )
    os.makedirs(outdir, exist_ok=True)


def run_evaluation(args: ap.ArgumentParser):
    import anndata as ad
    import polars as pl
    from pdex import parallel_differential_expression

    from cell_eval import (
        MetricPipeline,
        PerturbationAnndataPair,
        initialize_de_comparison,
    )

    build_outdir(args.outdir)

    adata_real = ad.read_h5ad(args.adata_real)
    adata_pred = ad.read_h5ad(args.adata_pred)

    if not args.de_real:
        logger.info("Computing DE for real data")
        de_real = parallel_differential_expression(
            adata=adata_real,
            reference=args.control_pert,
            groupby_key=args.pert_col,
            metric=args.de_method,
            num_workers=args.num_threads,
            batch_size=args.batch_size,
            as_polars=True,
        )
        de_real.write_csv(os.path.join(args.outdir, "de_real.csv"))
    else:
        de_real = pl.read_csv(
            args.de_real,
            schema_overrides={
                "target": pl.Utf8,
                "feature": pl.Utf8,
            },
        )

    if not args.de_pred:
        logger.info("Computing DE for predicted data")
        de_pred = parallel_differential_expression(
            adata=adata_pred,
            reference=args.control_pert,
            groupby_key=args.pert_col,
            metric=args.de_method,
            num_workers=args.num_threads,
            batch_size=args.batch_size,
            as_polars=True,
        )
        de_pred.write_csv(os.path.join(args.outdir, "de_pred.csv"))
    else:
        de_pred = pl.read_csv(
            args.de_pred,
            schema_overrides={
                "target": pl.Utf8,
                "feature": pl.Utf8,
            },
        )

    data_de = initialize_de_comparison(
        real=de_real,
        pred=de_pred,
        target_col="target",
    )

    data_anndata = PerturbationAnndataPair(
        real=adata_real,
        pred=adata_pred,
        control_pert=args.control_pert,
        pert_col=args.pert_col,
    )

    pipeline = MetricPipeline(
        profile="full",
    )
    pipeline.compute_de_metrics(data_de)
    pipeline.compute_anndata_metrics(data_anndata)
    pipeline.get_results().write_csv(os.path.join(args.outdir, "results.csv"))
