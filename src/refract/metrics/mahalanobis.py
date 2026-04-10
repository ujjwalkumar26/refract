"""Mahalanobis distance-based similarity metric.

A geometry-aware metric that accounts for correlations between embedding
dimensions. Particularly effective in dense, anisotropic embedding spaces
where cosine similarity loses discriminative power.

Requires fitting on the candidate corpus before use.
"""

from __future__ import annotations

import numpy as np

from refract.metrics.base import BaseMetric


class MahalanobisMetric(BaseMetric):
    """Mahalanobis distance-based similarity.

    Computes similarity using the Mahalanobis distance, which accounts
    for the covariance structure of the embedding space. This makes it
    more discriminative than Euclidean distance in anisotropic spaces.

    Must call ``fit(corpus_vectors)`` before scoring. The fusion engine
    handles this automatically.

    Similarity = 1 / (1 + mahalanobis_distance).
    """

    name = "mahalanobis"
    requires_fitting = True

    def __init__(self, regularization: float = 1e-5) -> None:
        """Initialize Mahalanobis metric.

        Args:
            regularization: Regularization term added to the covariance
                matrix diagonal for numerical stability.
        """
        self._cov_inv: np.ndarray | None = None
        self._regularization = regularization
        self._fitted = False

    def fit(self, corpus_vectors: np.ndarray) -> None:
        """Compute inverse covariance matrix from corpus.

        For large corpora (>5000 vectors), subsamples 3000 vectors
        for computational efficiency.

        Args:
            corpus_vectors: Matrix of shape ``(n, dim)`` with corpus embeddings.

        Raises:
            ValueError: If corpus has fewer than 2 vectors.
        """
        if len(corpus_vectors) < 2:
            raise ValueError("Mahalanobis requires at least 2 corpus vectors.")

        # Subsample for large corpora
        vecs = corpus_vectors
        if len(vecs) > 5000:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(vecs), size=3000, replace=False)
            vecs = vecs[indices]

        # Compute covariance with regularization
        cov = np.cov(vecs.T)
        if cov.ndim == 0:
            # Single-dimension edge case
            cov = np.array([[cov]])
        cov += np.eye(cov.shape[0]) * self._regularization

        # Use pseudoinverse for numerical stability
        self._cov_inv = np.linalg.pinv(cov)
        self._fitted = True

    def score(self, query_vec: np.ndarray, candidate_vec: np.ndarray) -> float:
        """Compute Mahalanobis similarity between two vectors.

        Args:
            query_vec: Query vector of shape ``(dim,)``.
            candidate_vec: Candidate vector of shape ``(dim,)``.

        Returns:
            Similarity score in (0, 1].

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if not self._fitted or self._cov_inv is None:
            raise RuntimeError(
                "MahalanobisMetric.fit(corpus) must be called before scoring. "
                "This is normally handled automatically by refract.search()."
            )
        diff = query_vec - candidate_vec
        dist_sq = float(diff @ self._cov_inv @ diff)
        # Clamp negative values from numerical noise
        dist = np.sqrt(max(dist_sq, 0.0))
        return 1.0 / (1.0 + dist)

    def batch_score(self, query_vec: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Vectorized Mahalanobis similarity against all candidates.

        Uses ``np.einsum`` for efficient batch computation.

        Args:
            query_vec: Query vector of shape ``(dim,)``.
            candidates: Candidate matrix of shape ``(n, dim)``.

        Returns:
            Array of shape ``(n,)`` with similarity scores in (0, 1].

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if not self._fitted or self._cov_inv is None:
            raise RuntimeError(
                "MahalanobisMetric.fit(corpus) must be called before scoring."
            )
        diffs = candidates - query_vec[np.newaxis, :]
        # (n, dim) @ (dim, dim) → (n, dim), then element-wise multiply and sum
        # Equivalent to: [diff @ cov_inv @ diff for each diff]
        transformed = diffs @ self._cov_inv
        dists_sq = np.einsum("ij,ij->i", transformed, diffs)
        # Clamp numerical noise
        dists = np.sqrt(np.maximum(dists_sq, 0.0))
        return (1.0 / (1.0 + dists)).astype(np.float64)
