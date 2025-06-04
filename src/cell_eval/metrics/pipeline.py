"""Pipeline for computing metrics."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import polars as pl
import numpy as np
from cell_eval.metrics.registry import MetricRegistry, MetricType, registry
from cell_eval.metrics.types import DEComparison, PerturbationAnndataPair

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a metric computation."""

    name: str
    value: float | str
    celltype: Optional[str] = None
    perturbation: Optional[str] = None

    def to_dict(self) -> Dict[str, float | str]:
        """Convert result to dictionary."""
        return {
            "celltype": self.celltype,
            "perturbation": self.perturbation,
            "metric": self.name,
            "value": self.value,
        }


class MetricPipeline:
    """Pipeline for computing metrics."""

    def __init__(self) -> None:
        """Initialize pipeline."""
        self._metrics: List[str] = []
        self._results: List[MetricResult] = []

    def add_metrics(self, metrics: List[str]) -> None:
        """Add metrics to pipeline."""
        self._metrics.extend(metrics)

    def compute_de_metrics(
        self, data: DEComparison, celltype: Optional[str] = None
    ) -> None:
        """Compute DE metrics."""
        for name in self._metrics:
            try:
                logger.info(f"Computing metric '{name}'")
                value = registry.compute(name, data)
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

    def compute_perturbation_metrics(
        self, data: PerturbationAnndataPair, celltype: Optional[str] = None
    ) -> None:
        """Compute perturbation metrics."""
        for name in self._metrics:
            try:
                logger.info(f"Computing metric '{name}'")
                value = registry.compute(name, data)
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
