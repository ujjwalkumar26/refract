"""Main search API — the core entry point for refract.

Provides ``search()`` and ``search_batch()`` functions that wire together
analysis, routing, and fusion into a single call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from refract.analysis.query_analyzer import analyze_query
from refract.analysis.space_analyzer import analyze_space
from refract.fusion.weighted import fuse
from refract.metrics.base import BaseMetric
from refract.metrics.bm25 import BM25Metric
from refract.metrics.cosine import CosineMetric
from refract.metrics.euclidean import EuclideanMetric
from refract.metrics.mahalanobis import MahalanobisMetric
from refract.routing.heuristic import HeuristicRouter
from refract.types import SearchResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from refract.embedders.base import BaseEmbedder
    from refract.routing.base import BaseRouter


def _build_tfidf_vectors(texts: list[str], dim: int = 128) -> np.ndarray:
    """Build simple TF-IDF vectors as a lightweight fallback embedder.

    Used when no embedder is provided and corpus is text-only.
    This uses scikit-learn's TfidfVectorizer with SVD truncation.

    Args:
        texts: List of text strings.
        dim: Target dimensionality (truncated via SVD).

    Returns:
        Matrix of shape ``(len(texts), dim)`` with TF-IDF vectors.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    actual_dim = min(dim, len(texts) - 1, 1000)
    if actual_dim < 1:
        actual_dim = 1

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    if tfidf_matrix.shape[1] > actual_dim:
        svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        vectors = svd.fit_transform(tfidf_matrix)
    else:
        vectors = tfidf_matrix.toarray()

    # L2 normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    vectors = vectors / norms

    return vectors.astype(np.float64)


def _resolve_metrics(
    metrics_arg: list[Union[str, BaseMetric]] | None,
    corpus_texts: list[str] | None,
    corpus_vecs: np.ndarray,
) -> dict[str, BaseMetric]:
    """Resolve metric specifications into concrete metric instances.

    Args:
        metrics_arg: User-specified metrics (names or instances).
        corpus_texts: Corpus text (needed for BM25).
        corpus_vecs: Corpus vectors (needed for Mahalanobis fitting).

    Returns:
        Dictionary mapping metric name → fitted metric instance.
    """
    if metrics_arg is None:
        # Default metrics
        metric_instances: dict[str, BaseMetric] = {
            "cosine": CosineMetric(),
            "euclidean": EuclideanMetric(),
        }
        # Add BM25 if text is available
        if corpus_texts is not None:
            bm25 = BM25Metric()
            bm25.fit_text(corpus_texts)
            metric_instances["bm25"] = bm25
        # Add Mahalanobis if we have enough candidates
        if len(corpus_vecs) >= 3:
            maha = MahalanobisMetric()
            maha.fit(corpus_vecs)
            metric_instances["mahalanobis"] = maha
        return metric_instances

    # User-specified metrics
    metric_instances = {}
    for m in metrics_arg:
        if isinstance(m, str):
            if m == "cosine":
                metric_instances[m] = CosineMetric()
            elif m == "euclidean":
                metric_instances[m] = EuclideanMetric()
            elif m == "mahalanobis":
                maha = MahalanobisMetric()
                maha.fit(corpus_vecs)
                metric_instances[m] = maha
            elif m == "bm25":
                if corpus_texts is not None:
                    bm25 = BM25Metric()
                    bm25.fit_text(corpus_texts)
                    metric_instances[m] = bm25
            else:
                raise ValueError(
                    f"Unknown metric name: {m!r}. "
                    f"Built-in: cosine, euclidean, mahalanobis, bm25. "
                    f"Pass a BaseMetric instance for custom metrics."
                )
        elif isinstance(m, BaseMetric):
            if m.requires_fitting:
                m.fit(corpus_vecs)
            metric_instances[m.name] = m
        else:
            raise TypeError(f"Metric must be a string or BaseMetric instance, got {type(m)}")

    return metric_instances


def search(
    query: Union[str, np.ndarray],
    corpus: Union[Sequence[str], np.ndarray],
    *,
    top_k: int = 10,
    embedder: BaseEmbedder | None = None,
    router: BaseRouter | None = None,
    metrics: list[Union[str, BaseMetric]] | None = None,
    return_provenance: bool = True,
) -> list[SearchResult]:
    """Search a corpus for the most relevant results to a query.

    This is the main entry point for refract. It automatically handles:
    - Embedding text (if an embedder is provided, or using TF-IDF fallback)
    - Analyzing query type and search space geometry
    - Routing metric weights based on query and space characteristics
    - Fusing scores from multiple metrics
    - Building provenance traces for explainability

    Args:
        query: Query text string or pre-computed query vector.
        corpus: List of document strings or matrix of pre-computed vectors.
        top_k: Number of results to return. Defaults to 10.
        embedder: Optional embedding provider. If None and corpus is text,
            uses TF-IDF as a lightweight fallback.
        router: Optional custom router. Defaults to HeuristicRouter.
        metrics: Optional list of metrics to use (names or instances).
            Defaults to all applicable built-in metrics.
        return_provenance: Whether to include provenance traces (default True).

    Returns:
        List of SearchResult objects, sorted by score descending.

    Examples:
        Minimal — text query against text corpus:

        >>> results = refract.search("how to sort a list", docs)

        With a custom embedder:

        >>> results = refract.search("query", docs, embedder=my_embedder)

        With pre-computed vectors:

        >>> results = refract.search(query_vec, corpus_vecs)

        With custom metrics:

        >>> results = refract.search(query, docs, metrics=["cosine", "bm25"])
    """
    # ── Step 1: Resolve inputs ──────────────────────────────────────────────
    query_text: str | None = None
    query_vec: np.ndarray | None = None
    corpus_texts: list[str] | None = None
    corpus_vecs: np.ndarray

    if isinstance(query, str):
        query_text = query
    elif isinstance(query, np.ndarray):
        query_vec = query.astype(np.float64)
    else:
        raise TypeError(f"query must be str or np.ndarray, got {type(query)}")

    if isinstance(corpus, np.ndarray):
        corpus_vecs = corpus.astype(np.float64)
    elif isinstance(corpus, (list, tuple)):
        corpus_texts = list(corpus)
        if embedder is not None:
            corpus_vecs = embedder.embed(corpus_texts).astype(np.float64)
        else:
            corpus_vecs = _build_tfidf_vectors(corpus_texts)
    else:
        raise TypeError(f"corpus must be list[str] or np.ndarray, got {type(corpus)}")

    # ── Step 2: Embed query if needed ───────────────────────────────────────
    if query_text is not None and query_vec is None:
        if embedder is not None:
            query_vec = embedder.embed_one(query_text).astype(np.float64)
        else:
            # Use same TF-IDF space as corpus
            all_texts = [*list(corpus_texts or []), query_text]
            all_vecs = _build_tfidf_vectors(all_texts)
            # Re-assign corpus vectors from the same TF-IDF space
            corpus_vecs = all_vecs[:-1]
            query_vec = all_vecs[-1]

    # ── Step 3: Quick cosine scores for analysis ────────────────────────────
    cosine_scores: np.ndarray | None = None
    if query_vec is not None:
        cos_metric = CosineMetric()
        cosine_scores = cos_metric.batch_score(query_vec, corpus_vecs)

    # ── Step 4: Analyze query and space ─────────────────────────────────────
    query_profile = analyze_query(
        query_text=query_text,
        query_vector=query_vec,
        candidate_vectors=corpus_vecs,
        candidate_scores=cosine_scores,
    )

    space_profile = analyze_space(
        candidates=corpus_vecs,
        cosine_scores=cosine_scores,
    )

    # ── Step 5: Resolve metrics ─────────────────────────────────────────────
    metric_instances = _resolve_metrics(metrics, corpus_texts, corpus_vecs)

    # ── Step 6: Route ───────────────────────────────────────────────────────
    active_router = router if router is not None else HeuristicRouter()
    available_metric_names = list(metric_instances.keys())
    weights = active_router.route(query_profile, space_profile, available_metric_names)

    # ── Step 7: Fuse ────────────────────────────────────────────────────────
    fused = fuse(
        query_profile=query_profile,
        space_profile=space_profile,
        candidates=corpus_vecs,
        candidate_texts=corpus_texts,
        weights=weights,
        metrics=metric_instances,
        router_name=active_router.name,
    )

    # ── Step 8: Build SearchResult list ─────────────────────────────────────
    effective_k = min(top_k, len(fused))
    results: list[SearchResult] = []

    for idx, score, provenance in fused[:effective_k]:
        text = corpus_texts[idx] if corpus_texts is not None else None
        results.append(
            SearchResult(
                index=idx,
                text=text,
                vector=corpus_vecs[idx],
                score=score,
                provenance=provenance,
            )
        )

    return results


def search_batch(
    queries: Union[list[str], np.ndarray],
    corpus: Union[Sequence[str], np.ndarray],
    *,
    top_k: int = 10,
    embedder: BaseEmbedder | None = None,
    router: BaseRouter | None = None,
    metrics: list[Union[str, BaseMetric]] | None = None,
) -> list[list[SearchResult]]:
    """Search a corpus with multiple queries efficiently.

    Amortizes corpus embedding, space analysis, and metric fitting
    across all queries for better performance.

    Args:
        queries: List of query strings or matrix of query vectors.
        corpus: List of document strings or matrix of vectors.
        top_k: Number of results per query.
        embedder: Optional embedding provider.
        router: Optional custom router.
        metrics: Optional list of metrics to use.

    Returns:
        List of result lists, one per query.

    Example:
        >>> results = refract.search_batch(
        ...     ["query 1", "query 2", "query 3"],
        ...     documents,
        ... )
        >>> for query_results in results:
        ...     print(query_results[0].text)
    """
    # ── Resolve corpus once ─────────────────────────────────────────────────
    corpus_texts: list[str] | None = None
    corpus_vecs: np.ndarray

    if isinstance(corpus, np.ndarray):
        corpus_vecs = corpus.astype(np.float64)
    elif isinstance(corpus, (list, tuple)):
        corpus_texts = list(corpus)
        if embedder is not None:
            corpus_vecs = embedder.embed(corpus_texts).astype(np.float64)
        else:
            corpus_vecs = _build_tfidf_vectors(corpus_texts)
    else:
        raise TypeError(f"corpus must be list[str] or np.ndarray, got {type(corpus)}")

    # ── Resolve metrics once ────────────────────────────────────────────────
    metric_instances = _resolve_metrics(metrics, corpus_texts, corpus_vecs)

    # ── Analyze space once ──────────────────────────────────────────────────
    # We need a cosine scores sample for space analysis
    # Use centroid as a representative query
    centroid = corpus_vecs.mean(axis=0)
    cos_metric = CosineMetric()
    sample_scores = cos_metric.batch_score(centroid, corpus_vecs)
    space_profile = analyze_space(corpus_vecs, sample_scores)

    # ── Process queries ─────────────────────────────────────────────────────
    active_router = router if router is not None else HeuristicRouter()
    all_results: list[list[SearchResult]] = []

    # Resolve query vectors
    query_texts: list[str] | None = None
    query_vecs: np.ndarray | None = None

    if isinstance(queries, np.ndarray):
        query_vecs = queries.astype(np.float64)
    elif isinstance(queries, list):
        query_texts = queries
        if embedder is not None:
            query_vecs = embedder.embed(queries).astype(np.float64)
        else:
            # TF-IDF in same space as corpus
            all_texts = list(corpus_texts or []) + queries
            all_vecs = _build_tfidf_vectors(all_texts)
            corpus_vecs_new = all_vecs[: len(corpus_texts or [])]
            query_vecs = all_vecs[len(corpus_texts or []):]
            # Update corpus vecs to be in same TF-IDF space
            corpus_vecs = corpus_vecs_new
            # Re-fit metrics with new vectors
            metric_instances = _resolve_metrics(metrics, corpus_texts, corpus_vecs)

    n_queries = len(query_vecs) if query_vecs is not None else len(queries)

    for i in range(n_queries):
        q_text = query_texts[i] if query_texts is not None else None
        q_vec = query_vecs[i] if query_vecs is not None else None

        # Compute cosine scores for this query
        cosine_scores = None
        if q_vec is not None:
            cosine_scores = cos_metric.batch_score(q_vec, corpus_vecs)

        # Analyze query
        query_profile = analyze_query(
            query_text=q_text,
            query_vector=q_vec,
            candidate_vectors=corpus_vecs,
            candidate_scores=cosine_scores,
        )

        # Route
        available_metric_names = list(metric_instances.keys())
        weights = active_router.route(query_profile, space_profile, available_metric_names)

        # Fuse
        fused = fuse(
            query_profile=query_profile,
            space_profile=space_profile,
            candidates=corpus_vecs,
            candidate_texts=corpus_texts,
            weights=weights,
            metrics=metric_instances,
            router_name=active_router.name,
        )

        # Build results
        effective_k = min(top_k, len(fused))
        results: list[SearchResult] = []
        for idx, score, provenance in fused[:effective_k]:
            text = corpus_texts[idx] if corpus_texts is not None else None
            results.append(
                SearchResult(
                    index=idx,
                    text=text,
                    vector=corpus_vecs[idx],
                    score=score,
                    provenance=provenance,
                )
            )
        all_results.append(results)

    return all_results
