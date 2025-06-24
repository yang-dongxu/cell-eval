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
EXPECTED_GENE_DIM = 18080
MAX_CELL_DIM = 100000


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
        "--genes",
        type=str,
        help="CSV file containing expected gene names and order",
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
        "--expected-gene-dim",
        type=int,
        help=f"Expected gene dimension ({EXPECTED_GENE_DIM}) [default: %(default)s]. Set to -1 to disable.",
        default=EXPECTED_GENE_DIM,
    )
    parser.add_argument(
        "--max-cell-dim",
        type=int,
        help=f"Maximum cell dimension ({MAX_CELL_DIM}) [default: %(default)s]. Set to -1 to disable.",
        default=MAX_CELL_DIM,
    )
    parser.add_argument(
        "--skip-watermark",
        action="store_true",
        help="Skip watermarking the data",
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
    genes: str | None = None,
    max_cell_dim: int | None = MAX_CELL_DIM,
    exp_gene_dim: int | None = EXPECTED_GENE_DIM,
    watermark: bool = True,
):
    import polars as pl

    # Force anndata var to string
    adata.var.index = adata.var.index.astype(str)

    if pert_col not in adata.obs:
        raise ValueError(
            f"Provided perturbation column: {pert_col} missing from anndata: {adata.obs.columns}"
        )
    if celltype_col:
        if celltype_col not in adata.obs:
            raise ValueError(
                f"Provided celltype column: {celltype_col} missing from anndata: {adata.obs.columns}"
            )

    # Validate gene identity and ordering
    if genes:
        # Read in the genelist and cast to string
        genelist = pl.read_csv(genes, has_header=False).to_series(0).cast(str).to_list()

        # Check if expected dimension is provided and matches the length of the genelist
        if exp_gene_dim and len(genelist) != exp_gene_dim:
            logger.warning(
                f"Provided gene dimension: {len(genelist)} does not match expected gene dimension: {exp_gene_dim}."
            )
            logger.info(f"Setting expected gene dimension to {len(genelist)}")
            exp_gene_dim = len(genelist)
    else:
        genelist = adata.var_names.tolist()

    if adata.var_names.tolist() != genelist:
        missing_genes = set(genelist) - set(adata.var_names.tolist())
        extra_genes = set(adata.var_names.tolist()) - set(genelist)
        if len(missing_genes) == 0 and len(extra_genes) == 0:
            logger.warning(
                "Provided anndata contains all expected genes but they are out of order."
            )
            logger.info("Reordering genes...")
            adata = adata[:, np.array(genelist)]
        else:
            raise ValueError(
                "Provided gene list does not match anndata gene names:\n"
                f"Missing genes: {missing_genes}\n"
                f"Extra genes: {extra_genes}"
            )

    if exp_gene_dim and adata.shape[1] != exp_gene_dim:
        raise ValueError(
            f"Provided gene dimension: {adata.shape[1]} does not match expected gene dimension: {exp_gene_dim}"
        )

    if max_cell_dim and adata.shape[0] > max_cell_dim:
        raise ValueError(
            f"Provided cell dimension: {adata.shape[0]} exceeds maximum cell dimension: {max_cell_dim}"
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

    if watermark:
        logger.info("Noting prep pass")
        minimal.uns["prep-pass"] = True

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
        exp_gene_dim=args.expected_gene_dim if args.expected_gene_dim != -1 else None,
        max_cell_dim=args.max_cell_dim if args.max_cell_dim != -1 else None,
        genes=args.genes,
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
