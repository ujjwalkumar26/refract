"""Composite router — blend outputs from multiple routers.

Useful for combining heuristic rules with learned predictions,
or for ensembling multiple specialized routers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from refract.routing.base import BaseRouter

if TYPE_CHECKING:
    from refract.types import QueryProfile, SpaceProfile


class CompositeRouter(BaseRouter):
    """Composite router that blends outputs from multiple sub-routers.

    The final weights are a weighted average of each sub-router's output.

    Example:
        >>> from refract.routing import HeuristicRouter
        >>> composite = CompositeRouter([
        ...     (HeuristicRouter(), 0.7),
        ...     (my_learned_router, 0.3),
        ... ])
        >>> weights = composite.route(query_profile, space_profile, metrics)
    """

    name = "composite"

    def __init__(self, routers: list[tuple[BaseRouter, float]]) -> None:
        """Initialize composite router.

        Args:
            routers: List of (router, router_weight) tuples.
                Router weights must be non-negative and should sum to 1.0.

        Raises:
            ValueError: If no routers provided or weights sum to 0.
        """
        if not routers:
            raise ValueError("CompositeRouter requires at least one sub-router.")

        self.routers = routers
        total = sum(w for _, w in routers)
        if total < 1e-12:
            raise ValueError("Router weights must sum to a positive number.")

        # Normalize router weights
        self._normalized = [(r, w / total) for r, w in routers]

    def route(
        self,
        query_profile: QueryProfile,
        space_profile: SpaceProfile,
        available_metrics: list[str],
    ) -> dict[str, float]:
        """Blend metric weights from all sub-routers.

        Args:
            query_profile: Analyzed query characteristics.
            space_profile: Analyzed search space geometry.
            available_metrics: List of available metric names.

        Returns:
            Blended weights dictionary (sum = 1.0).
        """
        combined: dict[str, float] = {}

        for router, router_weight in self._normalized:
            sub_weights = router.route(query_profile, space_profile, available_metrics)
            for metric_name, metric_weight in sub_weights.items():
                combined[metric_name] = combined.get(metric_name, 0.0) + (
                    metric_weight * router_weight
                )

        # Normalize final weights
        total = sum(combined.values())
        if total < 1e-12:
            n = len(available_metrics)
            return {m: 1.0 / n for m in available_metrics} if n > 0 else {}

        return {k: v / total for k, v in combined.items()}

    def __repr__(self) -> str:
        sub = ", ".join(f"({r.name}, {w:.2f})" for r, w in self._normalized)
        return f"CompositeRouter([{sub}])"
