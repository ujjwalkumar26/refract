"""Weighted score fusion engine.

Combines scores from multiple similarity metrics using router-assigned
weights. Builds provenance traces for every result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from refract.metrics.bm25 import BM25Metric
from refract.types import MetricScore, Provenance, QueryProfile, SpaceProfile

if TYPE_CHECKING:
    from refract.metrics.base import BaseMetric


def fuse(
    query_profile: QueryProfile,
    space_profile: SpaceProfile,
    candidates: np.ndarray,
    candidate_texts: list[str] | None,
    weights: dict[str, float],
    metrics: dict[str, BaseMetric],
    router_name: str,
) -> list[tuple[int, float, Provenance]]:
    """Fuse scores from multiple metrics into a single ranked list.

    For each candidate, computes weighted scores from all active metrics
    and builds a provenance trace explaining the scoring.

    Metrics with weight below 0.01 are skipped for performance.

    Args:
        query_profile: Analyzed query profile.
        space_profile: Analyzed space profile.
        candidates: Candidate matrix of shape ``(n, dim)``.
        candidate_texts: Original texts (needed for BM25). Can be None.
        weights: Metric name → weight mapping from the router.
        metrics: Metric name → metric instance mapping.
        router_name: Name of the router that produced the weights.

    Returns:
        List of (candidate_index, final_score, provenance) tuples,
        sorted by final_score descending.
    """
    n_candidates = len(candidates)

    # Collect per-metric scores
    metric_results: dict[str, np.ndarray] = {}

    for metric_name, weight in weights.items():
        # Skip negligible weights
        if weight < 0.01:
            continue

        metric = metrics.get(metric_name)
        if metric is None:
            continue

        if metric.is_text_metric and isinstance(metric, BM25Metric):
            # BM25 uses text, not vectors
            if query_profile.raw is not None:
                scores = metric.batch_score_text(query_profile.raw)
            else:
                scores = np.zeros(n_candidates, dtype=np.float64)
        else:
            # Vector-based metric
            if query_profile.vector is not None:
                scores = metric.batch_score(query_profile.vector, candidates)
            else:
                scores = np.zeros(n_candidates, dtype=np.float64)

        metric_results[metric_name] = scores

    # Build results with provenance
    results: list[tuple[int, float, Provenance]] = []

    for idx in range(n_candidates):
        metric_scores: list[MetricScore] = []
        final_score = 0.0

        for metric_name, weight in weights.items():
            if metric_name in metric_results:
                raw_score = float(metric_results[metric_name][idx])
                weighted_score = raw_score * weight
                final_score += weighted_score
                metric_scores.append(
                    MetricScore(
                        metric_name=metric_name,
                        raw_score=raw_score,
                        weight=weight,
                        weighted_score=weighted_score,
                    )
                )

        # Sort metric scores by weighted contribution (descending)
        metric_scores.sort(key=lambda ms: ms.weighted_score, reverse=True)

        provenance = Provenance(
            metric_scores=metric_scores,
            router_name=router_name,
            query_type=query_profile.query_type,
            space_density=space_profile.density,
            final_score=final_score,
        )

        results.append((idx, final_score, provenance))

    # Sort by final score descending
    results.sort(key=lambda r: r[1], reverse=True)

    return results
