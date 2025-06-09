import logging
import multiprocessing as mp
import os
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from pdex import parallel_differential_expression

from ._pipeline import MetricPipeline
from ._types import PerturbationAnndataPair, initialize_de_comparison

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Evaluates benchmarking metrics of a predicted and real anndata object.
    """

    def __init__(
        self,
        adata_pred: ad.AnnData | str,
        adata_real: ad.AnnData | str,
        de_pred: pl.DataFrame | str | None = None,
        de_real: pl.DataFrame | str | None = None,
        control_pert: str = "non-targeting",
        pert_col: str = "target",
        de_method: str = "wilcoxon",
        num_threads: int = -1,
        batch_size: int = 100,
        outdir: str = "./cell-eval-outdir",
        allow_discrete: bool = False,
        prefix: str | None = None,
        pdex_kwargs: dict[str, Any] = {},
    ):
        if os.path.exists(outdir):
            logger.warning(
                f"Output directory {outdir} already exists, potential overwrite occurring"
            )
        os.makedirs(outdir, exist_ok=True)

        self.anndata_pair = _build_anndata_pair(
            real=adata_real,
            pred=adata_pred,
            control_pert=control_pert,
            pert_col=pert_col,
            allow_discrete=allow_discrete,
        )
        self.de_comparison = _build_de_comparison(
            anndata_pair=self.anndata_pair,
            de_pred=de_pred,
            de_real=de_real,
            de_method=de_method,
            num_threads=num_threads if num_threads != -1 else mp.cpu_count(),
            batch_size=batch_size,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs,
        )

        self.outdir = outdir
        self.prefix = prefix

    def compute(
        self,
        profile: Literal["full", "minimal", "de", "anndata"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        basename: str = "results.csv",
        write_csv: bool = True,
    ) -> pl.DataFrame:
        pipeline = MetricPipeline(
            profile=profile,
            metric_configs=metric_configs,
        )
        pipeline.compute_de_metrics(self.de_comparison)
        pipeline.compute_anndata_metrics(self.anndata_pair)
        results = pipeline.get_results()

        if write_csv:
            results.write_csv(
                os.path.join(
                    self.outdir,
                    f"{self.prefix}_{basename}" if self.prefix else basename,
                )
            )
        return results


def _build_anndata_pair(
    real: ad.AnnData | str,
    pred: ad.AnnData | str,
    control_pert: str,
    pert_col: str,
    allow_discrete: bool = False,
    n_cells: int = 100,
):
    if isinstance(real, str):
        logger.info(f"Reading real anndata from {real}")
        real = ad.read_h5ad(real)
    if isinstance(pred, str):
        logger.info(f"Reading pred anndata from {pred}")
        pred = ad.read_h5ad(pred)

    # Validate that the input is normalized and log-transformed
    _validate_normlog(adata=real, allow_discrete=allow_discrete, n_cells=n_cells)
    _validate_normlog(adata=pred, allow_discrete=allow_discrete, n_cells=n_cells)

    # Build the anndata pair
    return PerturbationAnndataPair(
        real=real, pred=pred, control_pert=control_pert, pert_col=pert_col
    )


def _validate_normlog(
    adata: ad.AnnData, n_cells: int = 100, allow_discrete: bool = False
):
    def suspected_discrete(x: np.ndarray, n_cells: int) -> bool:
        top_n = min(x.shape[0], n_cells)
        rowsum = x[:top_n].sum(axis=1)
        frac, _ = np.modf(rowsum)
        return np.all(frac == 0)

    if suspected_discrete(adata.X, n_cells):
        if allow_discrete:
            logger.warning(
                "Error: adata_pred appears not to be log-transformed. We expect normed+logged input"
                "If this is an error, rerun with `allow_discrete=True`"
            )
            return
        raise ValueError(
            "Error: adata_pred appears not to be log-transformed. We expect normed+logged input"
            "If this is an error, rerun with `allow_discrete=True`"
        )


def _build_de_comparison(
    anndata_pair: PerturbationAnndataPair | None = None,
    de_pred: pl.DataFrame | str | None = None,
    de_real: pl.DataFrame | str | None = None,
    de_method: str = "wilcoxon",
    num_threads: int = 1,
    batch_size: int = 100,
    outdir: str | None = None,
    prefix: str | None = None,
    pdex_kwargs: dict[str, Any] = {},
):
    return initialize_de_comparison(
        real=_load_or_build_de(
            mode="real",
            de_path=de_real,
            anndata_pair=anndata_pair,
            de_method=de_method,
            num_threads=num_threads,
            batch_size=batch_size,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs,
        ),
        pred=_load_or_build_de(
            mode="pred",
            de_path=de_pred,
            anndata_pair=anndata_pair,
            de_method=de_method,
            num_threads=num_threads,
            batch_size=batch_size,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs,
        ),
    )


def _load_or_build_de(
    mode: Literal["pred", "real"],
    de_path: pl.DataFrame | str | None = None,
    anndata_pair: PerturbationAnndataPair | None = None,
    de_method: str = "wilcoxon",
    num_threads: int = 1,
    batch_size: int = 100,
    outdir: str | None = None,
    prefix: str | None = None,
    pdex_kwargs: dict[str, Any] = {},
) -> pl.DataFrame:
    if de_path is None:
        if anndata_pair is None:
            raise ValueError("anndata_pair must be provided if de_path is not provided")
        logger.info(f"Computing DE for {mode} data")
        frame = parallel_differential_expression(
            adata=anndata_pair.real if mode == "real" else anndata_pair.pred,
            reference=anndata_pair.control_pert,
            groupby_key=anndata_pair.pert_col,
            metric=de_method,
            num_workers=num_threads,
            batch_size=batch_size,
            as_polars=True,
            **pdex_kwargs,
        )
        if outdir is not None:
            pathname = f"{mode}_de.csv" if not prefix else f"{prefix}_{mode}_de.csv"
            frame.write_csv(os.path.join(outdir, pathname))

        return frame  # type: ignore
    elif isinstance(de_path, str):
        logger.info(f"Reading {mode} DE results from {de_path}")
        if len(pdex_kwargs) > 0:
            logger.warn("pdex_kwargs are ignored when reading from a CSV file")
        return pl.read_csv(
            de_path,
            schema_overrides={
                "target": pl.Utf8,
                "feature": pl.Utf8,
            },
        )
    elif isinstance(de_path, pl.DataFrame):
        if len(pdex_kwargs) > 0:
            logger.warn("pdex_kwargs are ignored when reading from a CSV file")
        return de_path
    elif isinstance(de_path, pd.DataFrame):
        if len(pdex_kwargs) > 0:
            logger.warn("pdex_kwargs are ignored when reading from a CSV file")
        return pl.from_pandas(de_path)
    else:
        raise TypeError(f"Unexpected type for de_path: {type(de_path)}")
