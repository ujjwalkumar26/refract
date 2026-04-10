"""Benchmark harness for evaluating refract against baselines.

Runs search on a dataset, computes standard IR metrics, and optionally
compares against a vanilla cosine baseline.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Union

import numpy as np

from refract.benchmark.datasets import BeirDataset, CustomDataset
from refract.benchmark.eval_metrics import mrr, ndcg_at_k, recall_at_k
from refract.metrics.cosine import CosineMetric

if TYPE_CHECKING:
    from refract.embedders.base import BaseEmbedder
    from refract.routing.base import BaseRouter


@dataclasses.dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes:
        dataset_name: Name of the evaluated dataset.
        method: Method name ("refract" or "cosine_baseline").
        recall_at_1: Recall@1 averaged across queries.
        recall_at_5: Recall@5 averaged across queries.
        recall_at_10: Recall@10 averaged across queries.
        ndcg_at_10: NDCG@10 averaged across queries.
        mrr_score: MRR averaged across queries.
        avg_latency_ms: Average per-query latency in milliseconds.
        n_queries: Number of queries evaluated.
    """

    dataset_name: str
    method: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    mrr_score: float
    avg_latency_ms: float
    n_queries: int

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.method} on {self.dataset_name}: "
            f"NDCG@10={self.ndcg_at_10:.3f}, "
            f"Recall@10={self.recall_at_10:.3f}, "
            f"MRR={self.mrr_score:.3f}, "
            f"latency={self.avg_latency_ms:.1f}ms)"
        )


class BenchmarkHarness:
    """Harness for evaluating search quality.

    Runs refract.search() on a dataset with relevance judgments,
    computes standard IR metrics, and optionally compares against
    a vanilla cosine-similarity baseline.

    Example:
        >>> harness = BenchmarkHarness()
        >>> results = harness.run(
        ...     dataset=CustomDataset(...),
        ...     embedder=my_embedder,
        ...     compare_cosine_baseline=True,
        ... )
        >>> for r in results:
        ...     print(f"{r.method:20s} NDCG@10={r.ndcg_at_10:.3f}")
    """

    def run(
        self,
        dataset: Union[BeirDataset, CustomDataset],
        embedder: BaseEmbedder | None = None,
        router: BaseRouter | None = None,
        top_k: int = 10,
        compare_cosine_baseline: bool = True,
    ) -> list[BenchmarkResult]:
        """Run benchmark evaluation.

        Args:
            dataset: Dataset with queries, corpus, and relevance judgments.
            embedder: Embedding provider (uses TF-IDF fallback if None).
            router: Custom router (uses HeuristicRouter if None).
            top_k: Number of results to retrieve per query.
            compare_cosine_baseline: Whether to also run vanilla cosine.

        Returns:
            List of BenchmarkResult objects (refract + optional baseline).
        """
        # Normalize dataset
        ds = dataset.to_custom() if isinstance(dataset, BeirDataset) else dataset

        # Embed corpus
        if embedder is not None:
            corpus_vecs = embedder.embed(ds.corpus).astype(np.float64)
        else:
            from refract.search import _build_tfidf_vectors

            all_texts = ds.corpus + ds.queries
            all_vecs = _build_tfidf_vectors(all_texts)
            corpus_vecs = all_vecs[: len(ds.corpus)]
            query_vecs_tfidf = all_vecs[len(ds.corpus) :]

        # Embed queries
        if embedder is not None:
            query_vecs = embedder.embed(ds.queries).astype(np.float64)
        else:
            query_vecs = query_vecs_tfidf  # type: ignore[possibly-undefined]

        results: list[BenchmarkResult] = []

        # ── Run refract ─────────────────────────────────────────────────────

        refract_result = self._evaluate(
            method_name="refract",
            dataset_name=ds.name,
            queries=ds.queries,
            query_vecs=query_vecs,
            corpus=ds.corpus,
            corpus_vecs=corpus_vecs,
            relevance=ds.relevance,
            top_k=top_k,
            use_refract=True,
            embedder=embedder,
            router=router,
        )
        results.append(refract_result)

        # ── Run cosine baseline ─────────────────────────────────────────────
        if compare_cosine_baseline:
            baseline_result = self._evaluate(
                method_name="cosine_baseline",
                dataset_name=ds.name,
                queries=ds.queries,
                query_vecs=query_vecs,
                corpus=ds.corpus,
                corpus_vecs=corpus_vecs,
                relevance=ds.relevance,
                top_k=top_k,
                use_refract=False,
            )
            results.append(baseline_result)

        return results

    def _evaluate(
        self,
        method_name: str,
        dataset_name: str,
        queries: list[str],
        query_vecs: np.ndarray,
        corpus: list[str],
        corpus_vecs: np.ndarray,
        relevance: dict[int, set[int]],
        top_k: int,
        use_refract: bool,
        embedder: BaseEmbedder | None = None,
        router: BaseRouter | None = None,
    ) -> BenchmarkResult:
        """Evaluate a single method on the dataset.

        Args:
            method_name: Name for the result.
            dataset_name: Dataset name.
            queries: Query texts.
            query_vecs: Query vectors.
            corpus: Corpus texts.
            corpus_vecs: Corpus vectors.
            relevance: Query → relevant doc mapping.
            top_k: Results per query.
            use_refract: Use refract.search (True) or cosine only (False).
            embedder: Optional embedder.
            router: Optional router.

        Returns:
            BenchmarkResult with averaged metrics.
        """
        cos_metric = CosineMetric()
        all_recall_1: list[float] = []
        all_recall_5: list[float] = []
        all_recall_10: list[float] = []
        all_ndcg_10: list[float] = []
        all_mrr: list[float] = []
        total_time = 0.0

        for q_idx in range(len(queries)):
            if q_idx not in relevance or not relevance[q_idx]:
                continue

            relevant = relevance[q_idx]
            start = time.perf_counter()

            if use_refract:
                import refract as _refract

                search_results = _refract.search(
                    query=query_vecs[q_idx],
                    corpus=corpus_vecs,
                    top_k=top_k,
                    router=router,
                )
                retrieved = [r.index for r in search_results]
            else:
                # Vanilla cosine
                scores = cos_metric.batch_score(query_vecs[q_idx], corpus_vecs)
                top_indices = np.argsort(-scores)[:top_k]
                retrieved = top_indices.tolist()

            elapsed = time.perf_counter() - start
            total_time += elapsed

            all_recall_1.append(recall_at_k(retrieved, relevant, 1))
            all_recall_5.append(recall_at_k(retrieved, relevant, 5))
            all_recall_10.append(recall_at_k(retrieved, relevant, 10))
            all_ndcg_10.append(ndcg_at_k(retrieved, relevant, 10))
            all_mrr.append(mrr(retrieved, relevant))

        n_queries = len(all_recall_1)

        def avg(vals):
            return sum(vals) / len(vals) if vals else 0.0

        return BenchmarkResult(
            dataset_name=dataset_name,
            method=method_name,
            recall_at_1=avg(all_recall_1),
            recall_at_5=avg(all_recall_5),
            recall_at_10=avg(all_recall_10),
            ndcg_at_10=avg(all_ndcg_10),
            mrr_score=avg(all_mrr),
            avg_latency_ms=(total_time / max(n_queries, 1)) * 1000,
            n_queries=n_queries,
        )
