"""Abstract base class for similarity metrics.

All built-in and custom metrics must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseMetric(ABC):
    """Abstract base class for similarity metrics.

    A metric computes a similarity score between a query vector and one or
    more candidate vectors. Scores must be in the range [0, 1], where
    1.0 means identical and 0.0 means maximally dissimilar.

    Subclasses must implement:
        - ``name``: A unique string identifier for the metric.
        - ``score()``: Compute similarity between two vectors.

    Subclasses may override:
        - ``batch_score()``: Vectorized scoring for performance.
        - ``requires_fitting``: Whether the metric needs corpus-level fitting.
        - ``fit()``: Fit the metric on the candidate corpus (e.g., Mahalanobis).

    Example:
        >>> class MyMetric(BaseMetric):
        ...     name = "my_metric"
        ...     def score(self, query_vec, candidate_vec):
        ...         return float(np.dot(query_vec, candidate_vec))
    """

    #: Unique name for this metric. Must be set by subclasses.
    name: str = ""

    #: Whether this metric requires `.fit(corpus)` before scoring.
    requires_fitting: bool = False

    #: Whether this metric operates on raw text instead of vectors.
    is_text_metric: bool = False

    @abstractmethod
    def score(self, query_vec: np.ndarray, candidate_vec: np.ndarray) -> float:
        """Compute similarity between two vectors.

        Args:
            query_vec: The query embedding vector.
            candidate_vec: A single candidate embedding vector.

        Returns:
            Similarity score in [0, 1]. Higher means more similar.
        """

    def batch_score(self, query_vec: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Score query against all candidates.

        Default implementation loops over ``score()``. Override this method
        with a vectorized implementation for better performance.

        Args:
            query_vec: The query embedding vector of shape ``(dim,)``.
            candidates: Candidate matrix of shape ``(n, dim)``.

        Returns:
            Array of shape ``(n,)`` with similarity scores.
        """
        return np.array(
            [self.score(query_vec, c) for c in candidates],
            dtype=np.float64,
        )

    def fit(self, corpus_vectors: np.ndarray) -> None:
        """Fit the metric on corpus vectors.

        Only called if ``requires_fitting`` is True. The default
        implementation is a no-op.

        Args:
            corpus_vectors: Matrix of shape ``(n, dim)`` with corpus embeddings.
        """
        return  # no-op by default

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
