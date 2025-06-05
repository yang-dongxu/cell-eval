import logging
from typing import Literal

import numpy as np
import polars as pl

from .._types import DEComparison, MetricType, PerturbationAnndataPair
from ..metrics import MetricResult, metrics_registry

logger = logging.getLogger(__name__)


class MetricPipeline:
    """Pipeline for computing metrics."""

    def __init__(
        self,
        profile: Literal["full", "de", "anndata"] | None = "full",
        metric_configs: dict[str, dict[str, any]] | None = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            profile: Which set of metrics to compute ('full', 'de', 'anndata', or None)
            metric_configs: Dictionary mapping metric names to their configuration kwargs
        """
        self._metrics: list[str] = []
        self._results: list[MetricResult] = []
        self._metric_configs = metric_configs or {}

        match profile:
            case "full":
                self._metrics.extend(metrics_registry.list_metrics(MetricType.DE))
                self._metrics.extend(
                    metrics_registry.list_metrics(MetricType.ANNDATA_PAIR)
                )
            case "de":
                self._metrics.extend(metrics_registry.list_metrics(MetricType.DE))
            case "anndata":
                self._metrics.extend(
                    metrics_registry.list_metrics(MetricType.ANNDATA_PAIR)
                )
            case None:
                pass
            case _:
                raise ValueError(f"Unrecognized profile: {profile}")

        # Apply metric configurations
        for metric_name, config in self._metric_configs.items():
            if metric_name in metrics_registry.list_metrics():
                metrics_registry.update_metric_kwargs(metric_name, config)

    def add_metrics(
        self, metrics: list[str], configs: dict[str, dict[str, any]] | None = None
    ) -> None:
        """Add metrics to pipeline.

        Args:
            metrics: List of metric names to add
            configs: Optional dictionary mapping metric names to their configuration kwargs
        """
        self._metrics.extend(metrics)
        if configs:
            self._metric_configs.update(configs)
            for metric_name, config in configs.items():
                if metric_name in metrics_registry.list_metrics():
                    metrics_registry.update_metric_kwargs(metric_name, config)

    def _compute_metric(
        self,
        name: str,
        data: DEComparison | PerturbationAnndataPair,
    ):
        """Compute a specific metric."""
        try:
            logger.info(f"Computing metric '{name}'")
            # Get any runtime config for this metric
            runtime_config = self._metric_configs.get(name, {})
            value = metrics_registry.compute(name, data, kwargs=runtime_config)
            if isinstance(value, dict):
                # Add each perturbation result separately
                for pert, pert_value in value.items():
                    if isinstance(pert_value, dict):
                        for sub_name, value in pert_value.items():
                            self._results.append(
                                MetricResult(
                                    name=f"{name}_{sub_name}",
                                    value=value,
                                    perturbation=pert,
                                )
                            )
                    else:
                        self._results.append(
                            MetricResult(
                                name=name,
                                value=pert_value,
                                perturbation=pert,
                            )
                        )
            else:
                # Add single result to all perturbations
                for pert in data.get_perts(include_control=False):
                    self._results.append(
                        MetricResult(
                            name=name,
                            value=value,
                            perturbation=pert,
                        )
                    )
        except Exception as e:
            logger.error(f"Error computing metric '{name}': {e}")

    def compute_de_metrics(self, data: DEComparison) -> None:
        """Compute DE metrics."""
        for name in self._metrics:
            if name not in metrics_registry.list_metrics(MetricType.DE):
                continue
            self._compute_metric(name, data)

    def compute_anndata_metrics(
        self,
        data: PerturbationAnndataPair,
    ) -> None:
        """Compute perturbation metrics."""
        for name in self._metrics:
            if name not in metrics_registry.list_metrics(MetricType.ANNDATA_PAIR):
                continue
            self._compute_metric(name, data)

    def get_results(self) -> pl.DataFrame:
        """Get results as a DataFrame."""
        if not self._results:
            return pl.DataFrame()
        return pl.DataFrame([r.to_dict() for r in self._results]).pivot(
            index="perturbation",
            on="metric",
            values="value",
        )

    def get_summary_stats(self) -> pl.DataFrame:
        """Get summary statistics for results."""
        if not self._results:
            return pl.DataFrame()

        # Group by metric and compute statistics
        stats = []
        for name in set(r.name for r in self._results):
            values = [r.value for r in self._results if r.name == name]
            if not values:
                continue
            stats.append(
                {
                    "metric": name,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )

        if not stats:
            return pl.DataFrame()

        return pl.DataFrame(stats).set_index("metric")
