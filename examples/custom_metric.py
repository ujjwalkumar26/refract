"""Custom metric example — plug in your own similarity metric.

Shows how to implement BaseMetric, register it, and use it in search.

Run with: python examples/custom_metric.py
"""

import numpy as np

import refract
from refract.metrics import BaseMetric


# ── Define a custom metric ──────────────────────────────────────────────────

class WeightedDotProduct(BaseMetric):
    """A custom metric that computes a weighted dot product.

    Weights the first half of dimensions more heavily, useful when
    earlier dimensions carry more information (e.g., PCA-reduced vectors).
    """

    name = "weighted_dot"

    def __init__(self, emphasis: float = 2.0) -> None:
        """
        Args:
            emphasis: How much more to weight the first half of dimensions.
        """
        self.emphasis = emphasis
        self._weights: np.ndarray | None = None

    def score(self, query_vec: np.ndarray, candidate_vec: np.ndarray) -> float:
        """Compute weighted dot product similarity."""
        if self._weights is None or len(self._weights) != len(query_vec):
            dim = len(query_vec)
            self._weights = np.ones(dim)
            self._weights[: dim // 2] *= self.emphasis
            self._weights /= self._weights.sum()  # Normalize

        weighted_dot = float(np.dot(query_vec * self._weights, candidate_vec))
        # Map to [0, 1] via sigmoid-like transform
        return 1.0 / (1.0 + np.exp(-weighted_dot * 5))

    def batch_score(self, query_vec: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Vectorized version for performance."""
        if self._weights is None or len(self._weights) != len(query_vec):
            dim = len(query_vec)
            self._weights = np.ones(dim)
            self._weights[: dim // 2] *= self.emphasis
            self._weights /= self._weights.sum()

        weighted_query = query_vec * self._weights
        dots = candidates @ weighted_query
        return 1.0 / (1.0 + np.exp(-dots * 5))


# ── Use it in search ────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = [
        "Sort a Python list using the sorted() built-in.",
        "Neural networks learn representations of data.",
        "Retrieve relevant documents from a large corpus.",
        "Use cosine similarity to measure vector closeness.",
    ]

    # Search with default metrics + our custom metric
    results = refract.search(
        "how to sort data",
        docs,
        metrics=["cosine", "bm25", WeightedDotProduct(emphasis=3.0)],
    )

    print("=" * 60)
    print("refract with custom metric: WeightedDotProduct")
    print("=" * 60)

    for r in results:
        print(f"\n  {r.score:.3f}  {r.text}")
        for ms in r.provenance.metric_scores:
            print(f"         {ms.metric_name:15s}  raw={ms.raw_score:.3f}  w={ms.weight:.2f}")
