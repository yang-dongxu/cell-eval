"""Array metrics module."""

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
