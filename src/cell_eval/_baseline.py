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
