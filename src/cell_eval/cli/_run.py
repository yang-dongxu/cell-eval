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
    from cell_eval import MetricsEvaluator

    evaluator = MetricsEvaluator(
        adata_pred=args.adata_pred,
        adata_real=args.adata_real,
        de_pred=args.de_pred,
        de_real=args.de_real,
        control_pert=args.control_pert,
        pert_col=args.pert_col,
        de_method=args.de_method,
        num_threads=args.num_threads,
        batch_size=args.batch_size,
        outdir=args.outdir,
    )
    results = evaluator.compute(profile="full")
    results.write_csv(os.path.join(args.outdir, "results.csv"))
