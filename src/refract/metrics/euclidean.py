"""Euclidean distance-based similarity metric.

Converts L2 distance to a similarity score via 1 / (1 + distance),
giving a smooth mapping from [0, ∞) distance to (0, 1] similarity.
"""

from __future__ import annotations

import numpy as np

from refract.metrics.base import BaseMetric


class EuclideanMetric(BaseMetric):
    """Euclidean distance-based similarity.

    Similarity = 1 / (1 + ||q - c||₂). This maps Euclidean distance
    to a similarity score in (0, 1], where 1.0 means identical vectors.

    The batch_score method is fully vectorized.
    """

    name = "euclidean"

    def score(self, query_vec: np.ndarray, candidate_vec: np.ndarray) -> float:
        """Compute Euclidean similarity between two vectors.

        Args:
            query_vec: Query vector of shape ``(dim,)``.
            candidate_vec: Candidate vector of shape ``(dim,)``.

        Returns:
            Similarity score in (0, 1].
        """
        dist = float(np.linalg.norm(query_vec - candidate_vec))
        return 1.0 / (1.0 + dist)

    def batch_score(self, query_vec: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Vectorized Euclidean similarity against all candidates.

        Args:
            query_vec: Query vector of shape ``(dim,)``.
            candidates: Candidate matrix of shape ``(n, dim)``.

        Returns:
            Array of shape ``(n,)`` with similarity scores in (0, 1].
        """
        diffs = candidates - query_vec[np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=1)
        return (1.0 / (1.0 + dists)).astype(np.float64)
