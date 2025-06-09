import argparse as ap
import importlib.metadata

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

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
        default="target_name",
        help="Name of the column designated perturbations",
    )
    parser.add_argument(
        "-c",
        "--celltype-col",
        type=str,
        help="Name of the column designated celltype (optional)",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        type=int,
        help=f"Bit size to encode ({VALID_ENCODINGS})",
        default=32,
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
    pert_col: str = "target_name",
    celltype_col: str | None = None,
    encoding: int = 64,
):
    if pert_col not in adata.obs:
        raise ValueError(
            f"Provided perturbation column: {pert_col} missing from anndata: {adata.obs.columns}"
        )
    if celltype_col:
        if celltype_col not in adata.obs:
            raise ValueError(
                f"Provided perturbation column: {celltype_col} missing from anndata: {adata.obs.columns}"
            )
    if encoding not in VALID_ENCODINGS:
        raise ValueError(f"Encoding must be in {VALID_ENCODINGS}")

    match encoding:
        case 64:
            dtype = np.dtype(np.float64)
        case 32:
            dtype = np.dtype(np.float32)
        case 16:
            dtype = np.dtype(np.float16)

    new_x = (
        adata.X.astype(dtype)
        if issparse(adata.X)
        else csr_matrix(adata.X.astype(dtype))
    )
    new_obs = pd.DataFrame(
        {"target_name": adata.obs[pert_col].values},
        index=np.arange(adata.shape[0]).astype(str),
    )
    if celltype_col:
        new_obs["celltype"] = adata.obs[celltype_col].values
    new_var = pd.DataFrame(
        index=adata.var.index.values,
    )
    minimal = ad.AnnData(
        X=new_x,
        obs=new_obs,
        var=new_var,
    )
    return minimal


def run_prep(args: ap.Namespace):
    adata = ad.read(args.input)
    minimal = strip_anndata(
        adata,
        pert_col=args.pert_col,
        celltype_col=args.celltype_col,
        encoding=args.encoding,
    )
    # drop adata from memory
    del adata

    # write output
    minimal.write_h5ad(
        args.output if args.output else args.input.replace(".h5ad", ".prep.h5ad"),
        compression="gzip",
    )
