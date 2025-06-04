"""Array metrics module."""

from typing import Callable

import numpy as np
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
    return _generic_evaluation(data, pearsonr, use_delta=True)


@registry.register(
    name="mse",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean squared error of each perturbation from control.",
)
def mse(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean squared error of each perturbation from control."""
    return _generic_evaluation(data, skm.mean_squared_error, use_delta=False)


@registry.register(
    name="mae",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean absolute error of each perturbation from control.",
)
def mae(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean absolute error of each perturbation from control."""
    return _generic_evaluation(data, skm.mean_absolute_error, use_delta=False)


@registry.register(
    name="mse_delta",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean squared error of each perturbation-control delta.",
)
def mse_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean squared error of each perturbation-control delta."""
    return _generic_evaluation(data, skm.mean_squared_error, use_delta=True)


@registry.register(
    name="mae_delta",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean absolute error of each perturbation-control delta.",
)
def mae_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean absolute error of each perturbation-control delta."""
    return _generic_evaluation(data, skm.mean_absolute_error, use_delta=True)


def _generic_evaluation(
    data: PerturbationAnndataPair,
    func: Callable[[np.ndarray, np.ndarray], float],
    use_delta: bool = False,
) -> dict[str, float]:
    """Generic evaluation function for anndata pair."""
    res = {}
    for delta_array in data.iter_delta_arrays():
        if use_delta:
            x = delta_array.pert_pred.mean(axis=0) - delta_array.ctrl_pred.mean(axis=0)
            y = delta_array.pert_real.mean(axis=0) - delta_array.ctrl_real.mean(axis=0)
        else:
            x = delta_array.pert_pred.mean(axis=0)
            y = delta_array.pert_real.mean(axis=0)

        result = func(x, y)
        if isinstance(result, tuple):
            result = result[0]

        res[delta_array.pert] = float(result)

    return res
