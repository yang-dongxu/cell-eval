import logging
import os
from typing import Literal

import anndata as ad
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
        num_threads: int = 1,
        batch_size: int = 100,
        outdir: str = "./cell-eval-outdir",
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
        )
        self.de_comparison = _build_de_comparison(
            anndata_pair=self.anndata_pair,
            de_pred=de_pred,
            de_real=de_real,
            de_method=de_method,
            num_threads=num_threads,
            batch_size=batch_size,
            outdir=outdir,
        )

    def compute(
        self,
        profile: Literal["full", "minimal"] = "full",
    ) -> pl.DataFrame:
        pipeline = MetricPipeline(
            profile=profile,
        )
        pipeline.compute_de_metrics(self.de_comparison)
        pipeline.compute_anndata_metrics(self.anndata_pair)
        return pipeline.get_results()


def _build_anndata_pair(
    real: ad.AnnData | str,
    pred: ad.AnnData | str,
    control_pert: str,
    pert_col: str,
):
    if isinstance(real, str):
        logger.info(f"Reading real anndata from {real}")
        real = ad.read_h5ad(real)
    if isinstance(pred, str):
        logger.info(f"Reading pred anndata from {pred}")
        pred = ad.read_h5ad(pred)
    return PerturbationAnndataPair(
        real=real, pred=pred, control_pert=control_pert, pert_col=pert_col
    )


def _build_de_comparison(
    anndata_pair: PerturbationAnndataPair | None = None,
    de_pred: pl.DataFrame | str | None = None,
    de_real: pl.DataFrame | str | None = None,
    de_method: str = "wilcoxon",
    num_threads: int = 1,
    batch_size: int = 100,
    outdir: str | None = None,
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
        ),
        pred=_load_or_build_de(
            mode="pred",
            de_path=de_pred,
            anndata_pair=anndata_pair,
            de_method=de_method,
            num_threads=num_threads,
            batch_size=batch_size,
            outdir=outdir,
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
):
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
        )
        if outdir is not None:
            frame.write_csv(os.path.join(outdir, f"{mode}_de.csv"))
        return frame
    elif isinstance(de_path, str):
        logger.info(f"Reading {mode} DE results from {de_path}")
        return pl.read_csv(
            de_path,
            schema_overrides={
                "target": pl.Utf8,
                "feature": pl.Utf8,
            },
        )
    elif isinstance(de_path, pl.DataFrame):
        return de_path
