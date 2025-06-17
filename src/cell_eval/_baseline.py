import logging

import anndata as ad
import numpy as np
import polars as pl
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def build_base_mean_adata(
    adata: ad.AnnData | str,
    counts_df: pl.DataFrame | str | None = None,
    pert_col: str = "target_gene",
    control_pert: str = "non-targeting",
    as_delta: bool = False,
    output_path: str | None = None,
) -> ad.AnnData:
    adata = ad.read_h5ad(adata) if isinstance(adata, str) else adata
    counts = (
        _load_counts_df(
            counts_df=counts_df, pert_col=pert_col, control_pert=control_pert
        )
        if counts_df is not None
        else _build_counts_df_from_adata(
            adata=adata,
            pert_col=pert_col,
            control_pert=control_pert,
        )
    )
    baseline = _build_pert_baseline(
        adata=adata, pert_col=pert_col, control_pert=control_pert
    )

    obs = (
        counts.select([pl.col(pert_col).repeat_by("len")]).explode(pert_col).to_pandas()
    )
    obs.index = obs.index.astype(str).str.replace("^", "p.", regex=True)

    logger.info("Assembling baseline adata from perturbation mean")
    baseline_adata = ad.AnnData(
        X=np.full(
            (int(counts["len"].sum()), baseline.size),
            baseline,
        ),
        var=adata.var,
        obs=obs,
    )

    logger.info("Concatenating baseline adata with controls from original adata")
    baseline_adata = ad.concat(
        [baseline_adata, adata[adata.obs[pert_col] == control_pert]]
    )

    if output_path is not None:
        logger.info(f"Saving baseline data to {output_path}")
        baseline_adata.write_h5ad(output_path)

    return baseline_adata


def _load_counts_df(
    counts_df: pl.DataFrame | str,
    pert_col: str = "target_gene",
    control_pert: str = "non-targeting",
) -> pl.DataFrame:
    if isinstance(counts_df, str):
        logger.info(f"Loading counts from {counts_df}")
        counts_df = pl.read_csv(counts_df)

    if pert_col not in counts_df.columns:
        raise ValueError(
            f"Column '{pert_col}' not found in counts_df: {counts_df.columns}"
        )

    logger.info(f"Filtering out counts from {control_pert}")
    return counts_df.filter(
        pl.col(pert_col) != control_pert  # drop control pert
    )


def _build_counts_df_from_adata(
    adata: ad.AnnData,
    pert_col: str = "target_gene",
    control_pert: str = "non-targeting",
) -> pl.DataFrame:
    if pert_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{pert_col}' not found in adata.obs: {adata.obs.columns}"
        )
    if control_pert not in adata.obs[pert_col].unique():
        raise ValueError(
            f"Control pert '{control_pert}' not found in adata.obs[{pert_col}]: {adata.obs[pert_col].unique()}"
        )
    logger.info("Building counts DataFrame from adata")
    return (
        pl.DataFrame(adata.obs)
        .group_by(pert_col)
        .len()
        .filter(pl.col(pert_col) != control_pert)
    )


def _build_pert_baseline(
    adata: ad.AnnData,
    pert_col: str = "target_gene",
    control_pert: str = "non-targeting",
    as_delta: bool = False,
) -> NDArray[np.float64]:
    if pert_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{pert_col}' not found in adata.obs: {adata.obs.columns}"
        )
    unique_perts = adata.obs[pert_col].unique()
    if control_pert not in unique_perts:
        raise ValueError(
            f"Control pert '{control_pert}' not found in unique_perts: {unique_perts}"
        )

    logger.info("Building perturbation-level means")
    pert_means = (
        pl.DataFrame(adata.X)
        .with_columns(pl.Series(pert_col, adata.obs[pert_col]))
        .group_by(pert_col)
        .mean()
    )

    names = pert_means.drop_in_place(pert_col).to_numpy()
    pert_mask = names != control_pert
    pert_matrix = pert_means.to_numpy()

    if as_delta:
        logger.info("Calculating delta from control means")
        delta = pert_matrix[pert_mask] - pert_matrix[~pert_mask]

        logger.info("Calculating mean delta")
        mean_delta = delta.mean(axis=0)

        return mean_delta
    else:
        logger.info("Calculating mean of perturbation-level means")
        mean_pert = pert_matrix.mean(axis=0)

        return mean_pert


# def fun_baseline_cell_mean(
#     ad_train,
#     ad_control,
#     ad_test,
#     pert_col="target_gene",
#     control_pert="non-targeting",
#     random_seed=1,
#     n_control_cells=-1,
# ):
#     import anndata as ad
#     import numpy as np
#     import pandas as pd

#     """
#     Calculate the baseline cell mean for each target gene in the training set
#     by averaging the control cells.
#     """
#     cell_mean_df = pd.DataFrame(index=ad_train.var_names)
#     cell_mean_df["Control"] = fun_cell_mean(adsplit["Control"])
#     cell_mean_df["Training"] = fun_cell_mean(adsplit["Training"])
#     cell_mean_df["Delta"] = cell_mean_df["Training"] - cell_mean_df["Control"]

#     np.random.seed(random_seed)
#     control_df = ad_control.to_df().T
#     if n_control_cells > 0:
#         # randomly select n_control_cells from control cells
#         control_df = control_df.sample(n=n_control_cells, axis=1)

#     control_df.columns = [control_pert + "_" + c for c in control_df.columns]

#     # for each pert in Validation set, generate the same number of cells by adding Delta to
#     # randomly selected cells from Control control cells
#     pert_ncells = ad_test.obs[pert_col].value_counts()
#     pred_list = [control_df]
#     for pert in pert_ncells.index:
#         # non-random: replicate cell_mean_df['Training'] for each pert pert_ncells[pert] times
#         df1 = pd.concat([cell_mean_df["Training"]] * pert_ncells[pert], axis=1)
#         df1.columns = [f"{pert}_{i}" for i in range(pert_ncells[pert])]

#         pred_list.append(df1)

#     pred_df = pd.concat(pred_list, axis=1)
#     pred_adata = ad.AnnData(
#         X=pred_df.T.values,
#         obs=pd.DataFrame(index=pred_df.T.index),
#         var=pd.DataFrame(index=pred_df.index),
#     )
#     pred_adata.obs[pert_col] = pred_adata.obs.index.str.split("_").str[0]

#     return pred_adata


# pred_adata = fun_baseline_cell_mean(
#     adsplit["Training"],
#     adsplit["Control"],
#     adsplit["Validation"],
#     pert_col="target_gene",
#     control_pert="non-targeting",
#     random_seed=1,
#     n_control_cells=-1,
# )
