"""Core data types for refract.

All data contracts are defined here as dataclasses. These types flow through
the entire pipeline: analysis → routing → fusion → results.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Literal, Union

import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class QueryProfile:
    """Profile of a query, computed by the query analyzer.

    Attributes:
        raw: Original query text, if provided.
        vector: Query embedding vector, if available.
        query_type: Detected query category.
        token_count: Number of whitespace-delimited tokens in the query.
        embedding_norm: L2 norm of the query vector (0.0 if no vector).
        entropy: Shannon entropy of softmax(cosine_scores) across candidates.
            Low entropy = one candidate dominates = high confidence.
            High entropy = flat scores = ambiguous query.
    """

    raw: str | None
    vector: np.ndarray | None
    query_type: Literal["keyword", "natural_language", "code", "structured"]
    token_count: int
    embedding_norm: float
    entropy: float

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        vec_shape = self.vector.shape if self.vector is not None else None
        return (
            f"QueryProfile(query_type={self.query_type!r}, "
            f"token_count={self.token_count}, "
            f"embedding_norm={self.embedding_norm:.3f}, "
            f"entropy={self.entropy:.3f}, "
            f"vector_shape={vec_shape})"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SpaceProfile:
    """Profile of the candidate search space.

    Computed by the space analyzer to characterize the geometry of the
    embedding space. Used by routers to select metric weights.

    Attributes:
        n_candidates: Number of candidate vectors.
        embedding_dim: Dimensionality of the embedding space.
        variance: Mean variance across embedding dimensions.
        anisotropy: Ratio of largest to smallest eigenvalue of covariance matrix.
            Higher values indicate more anisotropic (cone-shaped) distributions.
        density: Categorical density classification.
        score_spread: Standard deviation of cosine scores across candidates.
            Low spread means cosine is not discriminating well.
    """

    n_candidates: int
    embedding_dim: int
    variance: float
    anisotropy: float
    density: Literal["sparse", "medium", "dense"]
    score_spread: float

    def __repr__(self) -> str:
        return (
            f"SpaceProfile(n={self.n_candidates}, dim={self.embedding_dim}, "
            f"density={self.density!r}, anisotropy={self.anisotropy:.2f}, "
            f"score_spread={self.score_spread:.4f})"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class MetricScore:
    """Score contribution from a single similarity metric.

    Attributes:
        metric_name: Name of the metric (e.g., "cosine", "bm25").
        raw_score: Unweighted similarity score in [0, 1].
        weight: Weight assigned to this metric by the router.
        weighted_score: raw_score * weight.
    """

    metric_name: str
    raw_score: float
    weight: float
    weighted_score: float

    def __repr__(self) -> str:
        return (
            f"MetricScore({self.metric_name!r}: "
            f"raw={self.raw_score:.3f}, w={self.weight:.3f}, "
            f"weighted={self.weighted_score:.3f})"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Provenance:
    """Explainability trace for a single search result.

    Every result from refract.search() carries a provenance object that
    explains exactly how the final score was computed.

    Attributes:
        metric_scores: Per-metric breakdown of scores and weights.
        router_name: Name of the router that determined weights.
        query_type: Detected query type.
        space_density: Detected space density category.
        final_score: The weighted sum of all metric scores.
    """

    metric_scores: list[MetricScore]
    router_name: str
    query_type: str
    space_density: str
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance to a plain dictionary."""
        return {
            "final_score": self.final_score,
            "router": self.router_name,
            "query_type": self.query_type,
            "space_density": self.space_density,
            "metrics": {
                ms.metric_name: {
                    "score": ms.raw_score,
                    "weight": ms.weight,
                    "weighted": ms.weighted_score,
                }
                for ms in self.metric_scores
            },
        }

    def __repr__(self) -> str:
        metrics_str = ", ".join(
            f"{m.metric_name}={m.raw_score:.3f}x{m.weight:.2f}"
            for m in self.metric_scores
        )
        return (
            f"Provenance(score={self.final_score:.3f}, "
            f"router={self.router_name!r}, "
            f"query_type={self.query_type!r}, "
            f"density={self.space_density!r}, "
            f"[{metrics_str}])"
        )


@dataclasses.dataclass(slots=True)
class SearchResult:
    """A single search result with score and provenance.

    Attributes:
        index: Index into the original corpus.
        text: Original text if the corpus was provided as strings.
        vector: The candidate's embedding vector.
        score: Final weighted similarity score.
        provenance: Full explainability trace.
    """

    index: int
    text: str | None
    vector: np.ndarray
    score: float
    provenance: Provenance

    def __repr__(self) -> str:
        text_preview = None
        if self.text is not None:
            text_preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return (
            f"SearchResult(idx={self.index}, score={self.score:.3f}, "
            f"text={text_preview!r})"
        )

    def __lt__(self, other: SearchResult) -> bool:
        """Enable sorting by score (descending)."""
        return self.score > other.score


# ── Type aliases for public API ──────────────────────────────────────────────

#: A query can be a string (text) or a numpy array (pre-computed embedding).
QueryInput = Union[str, np.ndarray]

#: A corpus can be a list of strings or a 2D numpy array of embeddings.
CorpusInput = Union[Sequence[str], np.ndarray]
