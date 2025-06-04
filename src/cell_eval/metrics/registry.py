"""Registry module for metric computation."""

import enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
from .types import DEComparison, DeltaArrays


class MetricType(enum.Enum):
    """Types of metrics supported by the registry."""

    ARRAY = "array"
    DELTA = "delta"
    DE = "de"


@dataclass
class MetricInfo:
    """Information about a registered metric."""

    name: str
    type: MetricType
    func: Callable
    description: str
    is_class: bool = False


class Metric(Protocol):
    """Protocol for metric functions."""

    def __call__(self, data: Any) -> Union[float, Dict[str, float]]: ...


class MetricRegistry:
    """Registry for managing and accessing metrics."""

    def __init__(self) -> None:
        self.metrics: Dict[str, MetricInfo] = {}

    def register(
        self, name: str, metric_type: MetricType, description: str
    ) -> Callable[[Metric], Metric]:
        """
        Register a new metric.

        Args:
            name: Unique name for the metric
            metric_type: Type of metric being registered
            description: Description of what the metric computes

        Returns:
            Decorator function that registers the metric
        """

        def decorator(func: Metric) -> Metric:
            if name in self.metrics:
                raise ValueError(f"Metric '{name}' already registered")

            # Check if the metric is a class
            is_class = isinstance(func, type)

            self.metrics[name] = MetricInfo(
                name=name,
                type=metric_type,
                func=func,
                description=description,
                is_class=is_class,
            )
            return func

        return decorator

    def get_metric(self, name: str) -> MetricInfo:
        """Get information about a registered metric."""
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found in registry")
        return self.metrics[name]

    def list_metrics(self, metric_type: Optional[MetricType] = None) -> List[str]:
        """
        List registered metrics, optionally filtered by type.

        Args:
            metric_type: If provided, only list metrics of this type

        Returns:
            List of metric names
        """
        if metric_type is None:
            return list(self.metrics.keys())
        return [name for name, info in self.metrics.items() if info.type == metric_type]

    def compute(
        self,
        name: str,
        data: Union[DeltaArrays, DEComparison],
    ) -> Union[float, Dict[str, float]]:
        """
        Compute a metric on the provided data.

        Args:
            name: Name of the metric to compute
            data: Data to compute the metric on

        Returns:
            Metric result, either a single float or dictionary of values
        """
        metric = self.get_metric(name)
        if metric.is_class:
            # Instantiate the class before calling
            instance = metric.func()
            return instance(data)
        return metric.func(data)


# Global registry instance
registry = MetricRegistry()
