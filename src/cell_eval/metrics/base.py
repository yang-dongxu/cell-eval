from dataclasses import dataclass
from typing import Callable, Protocol

from .._types import DEComparison, MetricType, PerturbationAnndataPair


class Metric(Protocol):
    """Protocol for metric functions."""

    def __call__(
        self, data: PerturbationAnndataPair | DEComparison
    ) -> float | dict[str, float]: ...


@dataclass
class MetricResult:
    """Result of a metric computation."""

    name: str
    value: float | str
    celltype: str | None = None
    perturbation: str | None = None

    def to_dict(self) -> dict[str, float | str]:
        """Convert result to dictionary."""
        return {
            "celltype": self.celltype,
            "perturbation": self.perturbation,
            "metric": self.name,
            "value": self.value,
        }


@dataclass
class MetricInfo:
    """Information about a registered metric."""

    name: str
    type: MetricType
    func: Callable
    description: str
    is_class: bool = False
