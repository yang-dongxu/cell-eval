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
        self, profile: Literal["full", "de", "anndata"] | None = "full"
    ) -> None:
        """Initialize pipeline."""
        self._metrics: list[str] = []
        self._results: list[MetricResult] = []
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

    def add_metrics(self, metrics: list[str]) -> None:
        """Add metrics to pipeline."""
        self._metrics.extend(metrics)

    def compute_de_metrics(
        self, data: DEComparison, celltype: str | None = None
    ) -> None:
        """Compute DE metrics."""
        for name in self._metrics:
            if name not in metrics_registry.list_metrics(MetricType.DE):
                continue

            try:
                logger.info(f"Computing metric '{name}'")
                value = metrics_registry.compute(name, data)
                if isinstance(value, dict):
                    # Add each perturbation result separately
                    for pert, pert_value in value.items():
                        if isinstance(pert_value, dict):
                            for sub_name, value in pert_value.items():
                                self._results.append(
                                    MetricResult(
                                        name=f"{name}_{sub_name}",
                                        value=value,
                                        celltype=celltype,
                                        perturbation=pert,
                                    )
                                )
                        else:
                            self._results.append(
                                MetricResult(
                                    name=name,
                                    value=pert_value,
                                    celltype=celltype,
                                    perturbation=pert,
                                )
                            )
                else:
                    # Add single result to all perturbations
                    for pert in data.real.get_perts():
                        self._results.append(
                            MetricResult(
                                name=name,
                                value=value,
                                celltype=celltype,
                                perturbation=pert,
                            )
                        )
            except Exception as e:
                logger.error(f"Error computing metric '{name}': {e}")
                continue

    def compute_anndata_metrics(
        self, data: PerturbationAnndataPair, celltype: str | None = None
    ) -> None:
        """Compute perturbation metrics."""
        for name in self._metrics:
            if name not in metrics_registry.list_metrics(MetricType.ANNDATA_PAIR):
                continue

            try:
                logger.info(f"Computing metric '{name}'")
                value = metrics_registry.compute(name, data)
                if isinstance(value, dict):
                    # Add each perturbation result separately
                    for pert, pert_value in value.items():
                        if isinstance(pert_value, dict):
                            for sub_name, value in pert_value.items():
                                self._results.append(
                                    MetricResult(
                                        name=f"{name}_{sub_name}",
                                        value=value,
                                        celltype=celltype,
                                        perturbation=pert,
                                    )
                                )
                        else:
                            self._results.append(
                                MetricResult(
                                    name=name,
                                    value=pert_value,
                                    celltype=celltype,
                                    perturbation=pert,
                                )
                            )
                else:
                    # Add single result
                    self._results.append(
                        MetricResult(
                            name=name,
                            value=value,
                            celltype=celltype,
                        )
                    )
            except Exception as e:
                logger.error(f"Error computing metric '{name}': {e}")
                continue

    def get_results(self) -> pl.DataFrame:
        """Get results as a DataFrame."""
        if not self._results:
            return pl.DataFrame()
        return pl.DataFrame([r.to_dict() for r in self._results]).pivot(
            index=["celltype", "perturbation"],
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
