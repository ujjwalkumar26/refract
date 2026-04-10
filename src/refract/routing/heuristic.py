"""Heuristic router — rule-based metric weight assignment.

The default router. Uses hand-tuned rules based on query type and
space geometry to determine metric weights. No training required.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal

from refract.routing.base import BaseRouter

if TYPE_CHECKING:
    from refract.types import QueryProfile, SpaceProfile

# ── Type aliases ─────────────────────────────────────────────────────────────

_QueryType = Literal["keyword", "natural_language", "code", "structured"]
_Density = Literal["sparse", "medium", "dense"]
_RuleKey = tuple[_QueryType, _Density]

# ── Default rule table ───────────────────────────────────────────────────────

DEFAULT_RULES: dict[_RuleKey, dict[str, float]] = {
    # (query_type, density) → {metric: weight}
    ("keyword", "sparse"):           {"cosine": 0.50, "bm25": 0.40, "euclidean": 0.10},
    ("keyword", "medium"):           {"cosine": 0.45, "bm25": 0.30, "mahalanobis": 0.15, "euclidean": 0.10},
    ("keyword", "dense"):            {"cosine": 0.30, "bm25": 0.20, "mahalanobis": 0.40, "euclidean": 0.10},
    ("natural_language", "sparse"):  {"cosine": 0.65, "bm25": 0.25, "euclidean": 0.10},
    ("natural_language", "medium"):  {"cosine": 0.50, "bm25": 0.15, "mahalanobis": 0.25, "euclidean": 0.10},
    ("natural_language", "dense"):   {"cosine": 0.35, "bm25": 0.10, "mahalanobis": 0.45, "euclidean": 0.10},
    ("code", "sparse"):             {"cosine": 0.55, "bm25": 0.35, "euclidean": 0.10},
    ("code", "medium"):             {"cosine": 0.50, "bm25": 0.30, "mahalanobis": 0.10, "euclidean": 0.10},
    ("code", "dense"):              {"cosine": 0.40, "bm25": 0.25, "mahalanobis": 0.25, "euclidean": 0.10},
    ("structured", "sparse"):       {"bm25": 0.50, "cosine": 0.40, "euclidean": 0.10},
    ("structured", "medium"):       {"bm25": 0.45, "cosine": 0.35, "mahalanobis": 0.10, "euclidean": 0.10},
    ("structured", "dense"):        {"bm25": 0.40, "cosine": 0.25, "mahalanobis": 0.25, "euclidean": 0.10},
}

# ── Thresholds ───────────────────────────────────────────────────────────────

ENTROPY_HIGH_THRESHOLD = 2.5
ENTROPY_LOW_THRESHOLD = 1.0
SCORE_SPREAD_LOW_THRESHOLD = 0.02
ENTROPY_ADJUSTMENT_FACTOR = 0.20
SPREAD_ADJUSTMENT = 0.15


class HeuristicRouter(BaseRouter):
    """Rule-based metric weight router.

    Determines metric weights using a lookup table indexed by
    (query_type, space_density), with dynamic adjustments based on
    entropy and score spread.

    The rule table is exposed as ``rules`` and can be inspected or
    modified without subclassing.

    Adjustments applied after base lookup:
    1. **Entropy adjustment:** If query entropy is high (flat scores),
       the dominant metric's weight is reduced and redistributed.
    2. **Score spread adjustment:** If cosine score spread is very low
       (cosine can't discriminate), mahalanobis weight is boosted.

    Example:
        >>> router = HeuristicRouter()
        >>> weights = router.route(query_profile, space_profile, ["cosine", "bm25"])
        >>> print(weights)
        {'cosine': 0.65, 'bm25': 0.35}
    """

    name = "heuristic"

    def __init__(
        self,
        rules: dict[_RuleKey, dict[str, float]] | None = None,
        entropy_high: float = ENTROPY_HIGH_THRESHOLD,
        entropy_low: float = ENTROPY_LOW_THRESHOLD,
        spread_low: float = SCORE_SPREAD_LOW_THRESHOLD,
    ) -> None:
        """Initialize heuristic router.

        Args:
            rules: Custom rule table. If None, uses DEFAULT_RULES.
            entropy_high: Entropy threshold above which dominant metric
                weight is reduced.
            entropy_low: Entropy threshold below which base weights are kept.
            spread_low: Score spread threshold below which mahalanobis
                weight is boosted.
        """
        self.rules: dict[_RuleKey, dict[str, float]] = (
            copy.deepcopy(rules) if rules is not None else copy.deepcopy(DEFAULT_RULES)
        )
        self.entropy_high = entropy_high
        self.entropy_low = entropy_low
        self.spread_low = spread_low

    def route(
        self,
        query_profile: QueryProfile,
        space_profile: SpaceProfile,
        available_metrics: list[str],
    ) -> dict[str, float]:
        """Determine metric weights for this query + space combination.

        Args:
            query_profile: Analyzed query characteristics.
            space_profile: Analyzed search space geometry.
            available_metrics: List of available metric names.

        Returns:
            Dictionary mapping metric name to weight (sum = 1.0).
        """
        # Look up base weights
        key: _RuleKey = (query_profile.query_type, space_profile.density)  # type: ignore[assignment]
        base_weights = self.rules.get(key)

        if base_weights is None:
            # Fallback: equal weights across available metrics
            n = len(available_metrics)
            return {m: 1.0 / n for m in available_metrics} if n > 0 else {}

        # Filter to only available metrics
        weights = {m: base_weights.get(m, 0.0) for m in available_metrics}

        # Apply entropy adjustment
        weights = self._adjust_for_entropy(weights, query_profile.entropy)

        # Apply score spread adjustment
        weights = self._adjust_for_spread(weights, space_profile.score_spread)

        # Normalize to sum to 1.0
        weights = _normalize_weights(weights)

        return weights

    def _adjust_for_entropy(
        self, weights: dict[str, float], entropy: float
    ) -> dict[str, float]:
        """Adjust weights based on query entropy.

        If entropy is high (flat scores across candidates), the dominant
        metric's weight is reduced by ENTROPY_ADJUSTMENT_FACTOR and
        redistributed to other metrics.

        Args:
            weights: Current metric weights.
            entropy: Query entropy value.

        Returns:
            Adjusted weights dictionary.
        """
        if entropy <= self.entropy_high:
            return weights

        if not weights:
            return weights

        # Find the dominant metric
        dominant = max(weights, key=weights.get)  # type: ignore[arg-type]
        reduction = weights[dominant] * ENTROPY_ADJUSTMENT_FACTOR

        adjusted = dict(weights)
        adjusted[dominant] -= reduction

        # Redistribute to other metrics
        others = [m for m in adjusted if m != dominant and adjusted[m] > 0]
        if others:
            boost = reduction / len(others)
            for m in others:
                adjusted[m] += boost

        return adjusted

    def _adjust_for_spread(
        self, weights: dict[str, float], score_spread: float
    ) -> dict[str, float]:
        """Adjust weights based on cosine score spread.

        If score spread is very low (cosine can't discriminate),
        boost mahalanobis weight and reduce cosine.

        Args:
            weights: Current metric weights.
            score_spread: Standard deviation of cosine scores.

        Returns:
            Adjusted weights dictionary.
        """
        if score_spread >= self.spread_low:
            return weights

        adjusted = dict(weights)
        if "mahalanobis" in adjusted and "cosine" in adjusted:
            shift = min(SPREAD_ADJUSTMENT, adjusted["cosine"] * 0.5)
            adjusted["cosine"] -= shift
            adjusted["mahalanobis"] += shift

        return adjusted


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to sum to 1.0, clamping negatives to 0.

    Args:
        weights: Raw weights dictionary.

    Returns:
        Normalized weights with non-negative values summing to 1.0.
    """
    # Clamp negatives
    cleaned = {k: max(v, 0.0) for k, v in weights.items()}
    total = sum(cleaned.values())

    if total < 1e-12:
        # Fallback: equal weights
        n = len(cleaned)
        return {k: 1.0 / n for k in cleaned} if n > 0 else {}

    return {k: v / total for k, v in cleaned.items()}
