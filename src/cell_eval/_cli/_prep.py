import argparse as ap
import importlib.metadata
import logging

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from .._evaluator import _convert_to_normlog
from ._const import DEFAULT_CELLTYPE_COL, DEFAULT_PERT_COL

logger = logging.getLogger(__name__)

VALID_ENCODINGS = [64, 32]


def parse_args_prep(parser: ap.ArgumentParser):
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to h5ad to read",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to h5ad to write",
    )
    parser.add_argument(
        "-p",
        "--pert-col",
        type=str,
        default=DEFAULT_PERT_COL,
        help="Name of the column designated perturbations [default: %(default)s]",
    )
    parser.add_argument(
        "-c",
        "--celltype-col",
        type=str,
        help="Name of the column designated celltype (optional)",
    )
    parser.add_argument(
        "-P",
        "--output-pert-col",
        type=str,
        default=DEFAULT_PERT_COL,
        help="Name of the column designated perturbations in the output [default: %(default)s]",
    )
    parser.add_argument(
        "-C",
        "--output-celltype-col",
        type=str,
        default=DEFAULT_CELLTYPE_COL,
        help="Name of the column designated celltype in the output [default: %(default)s]",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        type=int,
        help=f"Bit size to encode ({VALID_ENCODINGS}) [default: %(default)s]",
        default=32,
    )
    parser.add_argument(
        "--allow-discrete",
        action="store_true",
        help="Bypass log normalization in case we incorrectly guess the data is discrete",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(
            version=importlib.metadata.version("cell_eval")
        ),
    )


def strip_anndata(
    adata: ad.AnnData,
    pert_col: str = "target_gene",
    celltype_col: str | None = None,
    output_pert_col: str = DEFAULT_PERT_COL,
    output_celltype_col: str = DEFAULT_CELLTYPE_COL,
    encoding: int = 64,
    allow_discrete: bool = False,
):
    if pert_col not in adata.obs:
        raise ValueError(
            f"Provided perturbation column: {pert_col} missing from anndata: {adata.obs.columns}"
        )
    if celltype_col:
        if celltype_col not in adata.obs:
            raise ValueError(
                f"Provided celltype column: {celltype_col} missing from anndata: {adata.obs.columns}"
            )
    if encoding not in VALID_ENCODINGS:
        raise ValueError(f"Encoding must be in {VALID_ENCODINGS}")

    dtype = np.dtype(np.float64)  # force bound
    match encoding:
        case 64:
            logger.info("Using 64-bit float encoding")
            dtype = np.dtype(np.float64)
        case 32:
            logger.info("Using 32-bit float encoding")
            dtype = np.dtype(np.float32)

    logger.info("Setting data to sparse if not already")
    new_x = (
        adata.X.astype(dtype)  # type: ignore
        if issparse(adata.X)
        else csr_matrix(adata.X.astype(dtype))  # type: ignore
    )

    logger.info("Simplifying obs dataframe")
    new_obs = pd.DataFrame(
        {output_pert_col: adata.obs[pert_col].values},
        index=np.arange(adata.shape[0]).astype(str),
    )
    if celltype_col:
        new_obs[output_celltype_col] = adata.obs[celltype_col].values

    logger.info("Simplifying var dataframe")
    new_var = pd.DataFrame(
        index=adata.var.index.values,
    )

    logger.info("Creating final minimal AnnData object")
    minimal = ad.AnnData(
        X=new_x,
        obs=new_obs,
        var=new_var,
    )

    logger.info("Applying normlog transformation if required")
    _convert_to_normlog(minimal, allow_discrete=allow_discrete)

    return minimal


def run_prep(args: ap.Namespace):
    logger.info("Reading input anndata")
    adata = ad.read_h5ad(args.input)

    logger.info("Preparing anndata")
    minimal = strip_anndata(
        adata,
        pert_col=args.pert_col,
        celltype_col=args.celltype_col,
        encoding=args.encoding,
        allow_discrete=args.allow_discrete,
    )
    # drop adata from memory
    del adata

    # write output
    outpath = args.output if args.output else args.input.replace(".h5ad", ".prep.h5ad")
    logger.info(f"Writing output to {outpath}")
    minimal.write_h5ad(
        outpath,
        compression="gzip",
    )
