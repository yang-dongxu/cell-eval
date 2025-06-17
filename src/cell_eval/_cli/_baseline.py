import argparse as ap
import importlib.metadata
import logging

logger = logging.getLogger(__name__)


def parse_args_baseline(parser: ap.ArgumentParser):
    """
    CLI for evaluation
    """
    parser.add_argument(
        "-a",
        "--adata",
        type=str,
        help="Path to the anndata",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--counts",
        type=str,
        help="Path to the perturbation cell counts",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path to save the baseline anndata",
        default="./baseline.h5ad",
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
        "--version",
        action="version",
        version="%(prog)s {version}".format(
            version=importlib.metadata.version("cell_eval")
        ),
    )


def run_baseline(args: ap.Namespace):
    from .._baseline import build_base_mean_adata

    build_base_mean_adata(
        adata=args.adata,
        counts_df=args.counts,
        control_pert=args.control_pert,
        pert_col=args.pert_col,
        output_path=args.output_path,
    )
