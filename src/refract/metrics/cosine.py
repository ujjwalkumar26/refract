"""Cosine similarity metric.

The most common similarity metric for embedding vectors. Computes the
cosine of the angle between two vectors: dot(q, c) / (||q|| * ||c||).
"""

from __future__ import annotations

import numpy as np

from refract.metrics.base import BaseMetric


class CosineMetric(BaseMetric):
    """Cosine similarity metric.

    Computes cosine similarity between vectors. Handles zero-norm vectors
    gracefully by returning 0.0.

    The batch_score method is fully vectorized for performance.
    """

    name = "cosine"

    def score(self, query_vec: np.ndarray, candidate_vec: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            query_vec: Query vector of shape ``(dim,)``.
            candidate_vec: Candidate vector of shape ``(dim,)``.

        Returns:
            Cosine similarity in [0, 1] (clamped from [-1, 1]).
        """
        norm_q = np.linalg.norm(query_vec)
        norm_c = np.linalg.norm(candidate_vec)
        if norm_q < 1e-12 or norm_c < 1e-12:
            return 0.0
        cos_sim = float(np.dot(query_vec, candidate_vec) / (norm_q * norm_c))
        # Map from [-1, 1] to [0, 1]
        return float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))

    def batch_score(self, query_vec: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity against all candidates.

        Args:
            query_vec: Query vector of shape ``(dim,)``.
            candidates: Candidate matrix of shape ``(n, dim)``.

        Returns:
            Array of shape ``(n,)`` with cosine similarities in [0, 1].
        """
        norm_q = np.linalg.norm(query_vec)
        if norm_q < 1e-12:
            return np.zeros(len(candidates), dtype=np.float64)

        norms_c = np.linalg.norm(candidates, axis=1)
        # Avoid division by zero
        safe_norms = np.where(norms_c < 1e-12, 1.0, norms_c)
        dots = candidates @ query_vec
        cos_sim = dots / (norm_q * safe_norms)
        # Zero out where candidate norm was zero
        cos_sim = np.where(norms_c < 1e-12, 0.0, cos_sim)
        # Map from [-1, 1] to [0, 1]
        return np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0).astype(np.float64)  # type: ignore[no-any-return]
