"""Standard information retrieval evaluation metrics.

Implements Recall@k, NDCG@k, and MRR for evaluating search quality.
"""

from __future__ import annotations

import math


def recall_at_k(
    retrieved: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Compute Recall@k.

    What fraction of relevant documents appear in the top-k results?

    Args:
        retrieved: Ordered list of retrieved document indices.
        relevant: Set of relevant document indices.
        k: Cutoff position.

    Returns:
        Recall score in [0, 1].
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def ndcg_at_k(
    retrieved: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at k (NDCG@k).

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        retrieved: Ordered list of retrieved document indices.
        relevant: Set of relevant document indices.
        k: Cutoff position.

    Returns:
        NDCG score in [0, 1].
    """
    if not relevant:
        return 0.0

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because positions are 1-indexed

    # Ideal DCG (all relevant documents at the top)
    ideal_k = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    if idcg < 1e-12:
        return 0.0

    return dcg / idcg


def mrr(
    retrieved: list[int],
    relevant: set[int],
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Returns the reciprocal of the rank of the first relevant document.

    Args:
        retrieved: Ordered list of retrieved document indices.
        relevant: Set of relevant document indices.

    Returns:
        MRR score in [0, 1]. Returns 0.0 if no relevant document found.
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0
