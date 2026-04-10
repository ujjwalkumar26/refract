"""Metric registry for discovering and managing similarity metrics.

Provides a central registry for looking up metrics by name. The default
registry is pre-populated with all built-in metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from refract.metrics.cosine import CosineMetric
from refract.metrics.euclidean import EuclideanMetric
from refract.metrics.mahalanobis import MahalanobisMetric

if TYPE_CHECKING:
    from refract.metrics.base import BaseMetric


class MetricRegistry:
    """Registry for similarity metrics.

    Provides register/lookup/list operations for metrics by name.
    Custom metrics can be registered at runtime.

    Example:
        >>> registry = MetricRegistry()
        >>> registry.register(CosineMetric())
        >>> metric = registry.get("cosine")
    """

    def __init__(self) -> None:
        self._metrics: dict[str, BaseMetric] = {}

    def register(self, metric: BaseMetric) -> None:
        """Register a metric instance.

        Args:
            metric: A metric instance to register.

        Raises:
            ValueError: If metric has no name.
        """
        if not metric.name:
            raise ValueError(f"Metric {metric!r} must have a non-empty `name` attribute.")
        self._metrics[metric.name] = metric

    def get(self, name: str) -> BaseMetric | None:
        """Look up a metric by name.

        Args:
            name: The metric name.

        Returns:
            The metric instance, or None if not found.
        """
        return self._metrics.get(name)

    def get_or_raise(self, name: str) -> BaseMetric:
        """Look up a metric by name, raising if not found.

        Args:
            name: The metric name.

        Returns:
            The metric instance.

        Raises:
            KeyError: If the metric is not registered.
        """
        metric = self._metrics.get(name)
        if metric is None:
            available = ", ".join(sorted(self._metrics.keys()))
            raise KeyError(
                f"Metric {name!r} not found. Available metrics: {available}"
            )
        return metric

    def list_available(self) -> list[str]:
        """List all registered metric names.

        Returns:
            Sorted list of metric name strings.
        """
        return sorted(self._metrics.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._metrics.keys()))
        return f"MetricRegistry([{names}])"


def create_default_registry() -> MetricRegistry:
    """Create a fresh registry with all built-in vector metrics.

    BM25 is not included because it requires corpus text at init time.
    It is registered lazily by the search engine when text is available.

    Returns:
        A MetricRegistry pre-populated with cosine, euclidean, mahalanobis.
    """
    registry = MetricRegistry()
    registry.register(CosineMetric())
    registry.register(EuclideanMetric())
    registry.register(MahalanobisMetric())
    return registry


#: The default global metric registry.
DEFAULT_REGISTRY: MetricRegistry = create_default_registry()
