"""Array metrics module."""

from typing import Callable

import numpy as np
import sklearn.metrics as skm
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from .._types import MetricType, PerturbationAnndataPair
from .registry import registry


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


@registry.register(
    name="discrimination_score",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Determines the similarity of each predicted perturbation to the real perturbation via normalized rank of cosine similarity",
)
def discrimination_score(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute perturbation discrimination score."""
    real_effects = np.vstack(
        [
            d.perturbation_effect(which="real", abs=True)
            for d in data.iter_delta_arrays()
        ]
    )
    pred_effects = np.vstack(
        [
            d.perturbation_effect(which="pred", abs=True)
            for d in data.iter_delta_arrays()
        ]
    )
    gene_names = data.real.var_names
    norm_ranks = {}
    for p_idx, p in enumerate(data.perts):
        include_mask = np.flatnonzero(gene_names != p)
        sim = cosine_similarity(
            real_effects[p_idx, include_mask].reshape(
                1, -1
            ),  # select real effect for current perturbation
            pred_effects[
                :, include_mask
            ],  # compare to all predicted effects across perturbations
        ).flatten()
        sorted_rev = np.argsort(sim)[::-1]
        p_index = np.flatnonzero(data.perts == p)[0]
        rank = np.flatnonzero(sorted_rev == p_index)[0]
        norm_rank = rank / data.perts.size
        norm_ranks[str(p)] = norm_rank
    return norm_ranks


def _generic_evaluation(
    data: PerturbationAnndataPair,
    func: Callable[[np.ndarray, np.ndarray], float],
    use_delta: bool = False,
) -> dict[str, float]:
    """Generic evaluation function for anndata pair."""
    res = {}
    for delta_array in data.iter_delta_arrays():
        if use_delta:
            x = delta_array.perturbation_effect(which="pred", abs=False)
            y = delta_array.perturbation_effect(which="real", abs=False)
        else:
            x = delta_array.pert_pred.mean(axis=0)
            y = delta_array.pert_real.mean(axis=0)

        result = func(x, y)
        if isinstance(result, tuple):
            result = result[0]

        res[delta_array.pert] = float(result)

    return res
