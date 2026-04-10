"""Abstract base class for routers.

Routers determine how to weight different similarity metrics based on
query characteristics and search space geometry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from refract.types import QueryProfile, SpaceProfile


class BaseRouter(ABC):
    """Abstract base class for metric weight routers.

    A router takes a query profile and a space profile, and returns a
    dictionary mapping metric names to weights. Weights must sum to 1.0.

    Subclasses must implement:
        - ``name``: A unique string identifier for the router.
        - ``route()``: Compute metric weights.

    Example:
        >>> class MyRouter(BaseRouter):
        ...     name = "my_router"
        ...     def route(self, query_profile, space_profile, available_metrics):
        ...         return {"cosine": 0.7, "bm25": 0.3}
    """

    #: Unique name for this router.
    name: str = ""

    @abstractmethod
    def route(
        self,
        query_profile: QueryProfile,
        space_profile: SpaceProfile,
        available_metrics: list[str],
    ) -> dict[str, float]:
        """Determine metric weights for this query and search space.

        Args:
            query_profile: Analyzed query characteristics.
            space_profile: Analyzed search space geometry.
            available_metrics: List of metric names that are available.

        Returns:
            Dictionary mapping metric name to weight.
            Weights must be non-negative and sum to 1.0.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
