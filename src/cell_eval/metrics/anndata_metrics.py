"""Array metrics module."""

import sklearn.metrics as skm
from scipy.stats import pearsonr

from .registry import MetricType, registry
from .types import PerturbationAnndataPair


@registry.register(
    name="pearson_delta",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Pearson correlation between mean differences from control",
)
def pearson_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute Pearson correlation between mean differences from control."""

    res = {}
    for delta_array in data.iter_delta_arrays():
        res[delta_array.pert] = float(
            pearsonr(
                delta_array.pert_pred.mean(axis=0) - delta_array.ctrl_pred.mean(axis=0),
                delta_array.pert_real.mean(axis=0) - delta_array.ctrl_real.mean(axis=0),
            )[0]
        )
    return res


@registry.register(
    name="mse",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean squared error of each perturbation from control.",
)
def mse(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean squared error of each perturbation from control."""
    res = {}
    for delta_array in data.iter_delta_arrays():
        res[delta_array.pert] = float(
            skm.mean_squared_error(
                delta_array.pert_pred.mean(axis=0),
                delta_array.pert_real.mean(axis=0),
            )
        )
    return res


@registry.register(
    name="mae",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean absolute error of each perturbation from control.",
)
def mae(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean absolute error of each perturbation from control."""
    res = {}
    for delta_array in data.iter_delta_arrays():
        res[delta_array.pert] = float(
            skm.mean_absolute_error(
                delta_array.pert_pred.mean(axis=0) - delta_array.ctrl_pred.mean(axis=0),
                delta_array.pert_real.mean(axis=0) - delta_array.ctrl_real.mean(axis=0),
            )
        )
    return res


@registry.register(
    name="mse_delta",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean squared error of each perturbation-control delta.",
)
def mse_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean squared error of each perturbation-control delta."""
    res = {}
    for delta_array in data.iter_delta_arrays():
        res[delta_array.pert] = float(
            skm.mean_squared_error(
                delta_array.pert_pred.mean(axis=0) - delta_array.ctrl_pred.mean(axis=0),
                delta_array.pert_real.mean(axis=0) - delta_array.ctrl_real.mean(axis=0),
            )
        )
    return res


@registry.register(
    name="mae_delta",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean absolute error of each perturbation-control delta.",
)
def mae_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean absolute error of each perturbation-control delta."""
    res = {}
    for delta_array in data.iter_delta_arrays():
        res[delta_array.pert] = float(
            skm.mean_absolute_error(
                delta_array.pert_pred.mean(axis=0) - delta_array.ctrl_pred.mean(axis=0),
                delta_array.pert_real.mean(axis=0) - delta_array.ctrl_real.mean(axis=0),
            )
        )
    return res
