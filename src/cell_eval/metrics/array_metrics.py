"""Array metrics module."""

from scipy.stats import pearsonr

from .registry import MetricType, registry
from .types import DeltaArrays


@registry.register(
    name="pearson_delta",
    metric_type=MetricType.DELTA,
    description="Pearson correlation between mean differences from control",
)
def pearson_delta(data: DeltaArrays) -> float:
    """Compute Pearson correlation between mean differences from control."""
    ctrl_pred = data.ctrl_pred if data.ctrl_pred is not None else data.ctrl_real
    return float(
        pearsonr(
            data.pert_pred.mean(axis=0) - ctrl_pred.mean(axis=0),
            data.pert_real.mean(axis=0) - data.ctrl_real.mean(axis=0),
        )[0]
    )
