"""Array metrics module."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from geomloss import SamplesLoss
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from .registry import MetricType, registry
from .types import ArrayPair, DeltaArrays


@registry.register(
    name="pearson_correlation",
    metric_type=MetricType.ARRAY,
    description="Pearson correlation between prediction and ground truth",
)
def pearson_correlation(data: ArrayPair) -> float:
    """Compute Pearson correlation between prediction and ground truth."""
    return float(pearsonr(data.pred.flatten(), data.real.flatten())[0])


@registry.register(
    name="mse",
    metric_type=MetricType.ARRAY,
    description="Mean squared error between prediction and ground truth",
)
def mse(data: ArrayPair) -> float:
    """Compute mean squared error between prediction and ground truth."""
    return float(mean_squared_error(data.pred, data.real))


@registry.register(
    name="pearson_delta",
    metric_type=MetricType.DELTA,
    description="Pearson correlation between mean differences from control",
)
def pearson_delta(data: DeltaArrays) -> float:
    """Compute Pearson correlation between mean differences from control."""
    ctrl_pred = data.ctrl_pred if data.ctrl_pred is not None else data.ctrl_real

    pred_delta = data.pert_pred - ctrl_pred
    real_delta = data.pert_real - data.ctrl_real

    return float(pearsonr(pred_delta.mean(0), real_delta.mean(0))[0])


@registry.register(
    name="wasserstein",
    metric_type=MetricType.ARRAY,
    description="Sinkhorn transport cost between point clouds",
)
@dataclass
class Wasserstein:
    """Compute Wasserstein distance between point clouds."""

    epsilon: float = 0.1

    def __call__(self, data: ArrayPair) -> float:
        """Compute transport cost between prediction and ground truth."""
        transport = SamplesLoss(
            loss="sinkhorn", p=2, blur=self.epsilon, backend="tensorized"
        )
        return float(transport(data.pred, data.real))


@registry.register(
    name="mmd",
    metric_type=MetricType.ARRAY,
    description="Maximum Mean Discrepancy with RBF kernel",
)
@dataclass
class MMD:
    """Compute Maximum Mean Discrepancy between distributions."""

    gammas: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if self.gammas is None:
            self.gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def _rbf_kernel(self, x: np.ndarray, y: np.ndarray, gamma: float) -> float:
        """Compute RBF kernel between x and y."""
        x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
        y_norm = np.sum(y**2, axis=1).reshape(1, -1)
        dist = x_norm + y_norm - 2 * np.dot(x, y.T)
        return np.exp(-gamma * dist)

    def _mmd_single_gamma(self, x: np.ndarray, y: np.ndarray, gamma: float) -> float:
        """Compute MMD for a single gamma value."""
        kxx = self._rbf_kernel(x, x, gamma)
        kyy = self._rbf_kernel(y, y, gamma)
        kxy = self._rbf_kernel(x, y, gamma)

        nx, ny = x.shape[0], y.shape[0]

        mmd = (
            np.sum(kxx - np.diag(np.diagonal(kxx))) / (nx * (nx - 1))
            + np.sum(kyy - np.diag(np.diagonal(kyy))) / (ny * (ny - 1))
            - 2 * np.sum(kxy) / (nx * ny)
        )
        return float(mmd)

    def __call__(self, data: ArrayPair) -> float:
        """Compute average MMD over multiple gamma values."""
        scores = []
        for gamma in self.gammas:
            try:
                score = self._mmd_single_gamma(data.pred, data.real, gamma)
                scores.append(score)
            except (ValueError, RuntimeError):
                continue

        return float(np.mean(scores)) if scores else 0.0
