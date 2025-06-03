"""Pipeline module for metric evaluation."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from .registry import MetricType, registry
from .types import (
    DEComparison,
    DeltaArrays,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a single metric computation."""

    name: str
    value: Union[float, Dict[str, float]]
    metric_type: MetricType
    perturbation: Optional[str] = None
    celltype: Optional[str] = None

    def to_dict(self) -> Dict[str, str | float]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type.value,
            "perturbation": self.perturbation,
            "celltype": self.celltype,
        }


@dataclass
class MetricPipeline:
    """Pipeline for computing and aggregating metrics."""

    metrics: List[str] = field(default_factory=list)
    celltypes: List[str] = field(default_factory=list)
    perturbations: List[str] = field(default_factory=list)
    _results: List[MetricResult] = field(default_factory=list, init=False)

    def add_metric(self, name: str) -> None:
        """Add a metric to the pipeline."""
        if name not in registry.metrics:
            raise ValueError(f"Metric '{name}' not found in registry")
        self.metrics.append(name)

    def add_metrics(self, names: Sequence[str]) -> None:
        """Add multiple metrics to the pipeline."""
        for name in names:
            self.add_metric(name)

    def compute_delta_metrics(
        self,
        data: DeltaArrays,
        celltype: str,
    ) -> None:
        """Compute delta-based metrics."""

        for name in self.metrics:
            metric_info = registry.get_metric(name)
            if metric_info.type != MetricType.DELTA:
                continue

            try:
                value = registry.compute(name, data)
                self._results.append(
                    MetricResult(
                        name=name,
                        value=value,
                        metric_type=MetricType.DELTA,
                        perturbation=data.pert,
                        celltype=celltype,
                    )
                )
            except Exception as e:
                logger.error(f"Error computing metric '{name}': {e}")

    def compute_de_metrics(
        self,
        data: DEComparison,
        celltype: Optional[str] = None,
    ) -> None:
        """Compute DE-based metrics."""
        for name in self.metrics:
            metric_info = registry.get_metric(name)
            if metric_info.type != MetricType.DE:
                continue

            try:
                value = registry.compute(name, data)
                if isinstance(value, dict):
                    # Add each perturbation result separately
                    for pert, pert_value in value.items():
                        self._results.append(
                            MetricResult(
                                name=name,
                                value=pert_value,
                                metric_type=MetricType.DE,
                                perturbation=pert,
                                celltype=celltype,
                            )
                        )
                else:
                    # Add single aggregated result to all perturbations
                    for pert in data.iter_perturbations():
                        self._results.append(
                            MetricResult(
                                name=name,
                                value=value,
                                metric_type=MetricType.DE,
                                perturbation=pert,
                                celltype=celltype,
                            )
                        )
            except Exception as e:
                logger.error(f"Error computing metric '{name}': {e}")

    def get_results(self) -> pd.DataFrame:
        """Get all metric results as a matrix-like DataFrame.

        Returns a DataFrame with columns:
        - celltype
        - perturbation
        - [metric1, metric2, ...]

        Where each row represents a unique celltype-perturbation combination.
        """
        results = pd.DataFrame([r.to_dict() for r in self._results])
        results = results.pivot(
            index=["celltype", "perturbation"], columns="name", values="value"
        ).reset_index()
        results.columns.name = None
        return results.sort_values(["celltype", "perturbation"])

    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics for each metric."""
        results = self.get_results()

        # Get metric columns
        metric_cols = [
            col for col in results.columns if col not in ["celltype", "perturbation"]
        ]

        if not metric_cols:
            return pd.DataFrame()

        # Calculate statistics
        stats = []
        for metric in metric_cols:
            values = results[metric].dropna()
            if len(values) == 0:
                continue

            stats.append(
                {
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "count": len(values),
                }
            )

        return pd.DataFrame(stats).set_index("metric")

    def clear_results(self) -> None:
        """Clear all computed results."""
        self._results = []
