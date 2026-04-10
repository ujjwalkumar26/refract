"""Integration tests for the main search API."""

from __future__ import annotations

import numpy as np
import pytest

import refract
from refract.routing.heuristic import HeuristicRouter


class TestSearchTextCorpus:
    """Test refract.search() with text corpus (TF-IDF fallback)."""

    def test_basic_search(self, sample_corpus_texts: list[str]) -> None:
        results = refract.search("how to sort in Python", sample_corpus_texts)
        assert len(results) > 0
        assert results[0].score > 0
        assert results[0].text is not None
        assert results[0].provenance is not None

    def test_top_k(self, sample_corpus_texts: list[str]) -> None:
        results = refract.search("Python sort", sample_corpus_texts, top_k=3)
        assert len(results) == 3

    def test_top_k_larger_than_corpus(self, sample_corpus_texts: list[str]) -> None:
        results = refract.search("test", sample_corpus_texts, top_k=100)
        assert len(results) == len(sample_corpus_texts)

    def test_provenance_has_metrics(self, sample_corpus_texts: list[str]) -> None:
        results = refract.search("sort a list", sample_corpus_texts)
        prov = results[0].provenance
        assert len(prov.metric_scores) > 0
        assert prov.router_name == "heuristic"
        assert prov.query_type in ("keyword", "natural_language", "code", "structured")

    def test_result_ordering(self, sample_corpus_texts: list[str]) -> None:
        results = refract.search("machine learning neural networks", sample_corpus_texts)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_custom_metrics(self, sample_corpus_texts: list[str]) -> None:
        results = refract.search(
            "sort list",
            sample_corpus_texts,
            metrics=["cosine", "bm25"],
        )
        metric_names = {ms.metric_name for ms in results[0].provenance.metric_scores}
        assert metric_names <= {"cosine", "bm25"}

    def test_custom_router(self, sample_corpus_texts: list[str]) -> None:
        router = HeuristicRouter()
        results = refract.search("test", sample_corpus_texts, router=router)
        assert len(results) > 0


class TestSearchVectorCorpus:
    """Test refract.search() with pre-computed vectors."""

    def test_vector_search(self, sample_query_vec: np.ndarray, sample_vectors: np.ndarray) -> None:
        results = refract.search(sample_query_vec, sample_vectors)
        assert len(results) > 0
        assert results[0].text is None  # No text for vector-only search
        assert results[0].score > 0

    def test_vector_provenance(
        self, sample_query_vec: np.ndarray, sample_vectors: np.ndarray
    ) -> None:
        results = refract.search(sample_query_vec, sample_vectors)
        prov = results[0].provenance
        assert prov is not None
        assert prov.final_score > 0


class TestSearchBatch:
    """Test refract.search_batch()."""

    def test_batch_text(self, sample_corpus_texts: list[str]) -> None:
        queries = ["sort a list", "neural networks", "binary search"]
        all_results = refract.search_batch(queries, sample_corpus_texts)
        assert len(all_results) == 3
        for results in all_results:
            assert len(results) > 0
            assert results[0].score > 0

    def test_batch_vectors(self, sample_vectors: np.ndarray) -> None:
        rng = np.random.default_rng(42)
        query_vecs = rng.standard_normal((3, 64))
        all_results = refract.search_batch(query_vecs, sample_vectors)
        assert len(all_results) == 3


class TestSearchEdgeCases:
    """Edge cases and error handling."""

    def test_single_doc_corpus(self) -> None:
        results = refract.search("test", ["single document"])
        assert len(results) == 1

    def test_two_doc_corpus(self) -> None:
        results = refract.search("hello", ["hello world", "goodbye world"])
        assert len(results) == 2

    def test_invalid_query_type(self) -> None:
        with pytest.raises(TypeError):
            refract.search(42, ["test"])  # type: ignore[arg-type]

    def test_invalid_corpus_type(self) -> None:
        with pytest.raises(TypeError):
            refract.search("test", 42)  # type: ignore[arg-type]


class TestVersion:
    def test_version_exists(self) -> None:
        assert refract.__version__
        assert isinstance(refract.__version__, str)
