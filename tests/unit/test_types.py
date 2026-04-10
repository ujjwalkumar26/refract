"""Tests for core types."""

from __future__ import annotations

import numpy as np

from refract.types import MetricScore, Provenance, QueryProfile, SearchResult, SpaceProfile


class TestQueryProfile:
    def test_creation(self) -> None:
        qp = QueryProfile(
            raw="test query",
            vector=np.array([1.0, 2.0]),
            query_type="keyword",
            token_count=2,
            embedding_norm=2.236,
            entropy=1.5,
        )
        assert qp.query_type == "keyword"
        assert qp.token_count == 2

    def test_repr(self) -> None:
        qp = QueryProfile(
            raw=None, vector=None, query_type="natural_language",
            token_count=0, embedding_norm=0.0, entropy=0.0,
        )
        assert "natural_language" in repr(qp)


class TestSpaceProfile:
    def test_creation(self) -> None:
        sp = SpaceProfile(
            n_candidates=100, embedding_dim=384,
            variance=0.1, anisotropy=5.0,
            density="medium", score_spread=0.05,
        )
        assert sp.density == "medium"
        assert sp.n_candidates == 100


class TestMetricScore:
    def test_creation(self) -> None:
        ms = MetricScore(
            metric_name="cosine", raw_score=0.9,
            weight=0.5, weighted_score=0.45,
        )
        assert ms.weighted_score == 0.45


class TestProvenance:
    def test_to_dict(self) -> None:
        ms = MetricScore("cosine", 0.9, 0.6, 0.54)
        prov = Provenance(
            metric_scores=[ms],
            router_name="heuristic",
            query_type="keyword",
            space_density="medium",
            final_score=0.54,
        )
        d = prov.to_dict()
        assert d["router"] == "heuristic"
        assert "cosine" in d["metrics"]
        assert d["metrics"]["cosine"]["score"] == 0.9


class TestSearchResult:
    def test_sorting(self) -> None:
        ms = MetricScore("cosine", 0.5, 1.0, 0.5)
        prov = Provenance([ms], "h", "nl", "medium", 0.5)
        r1 = SearchResult(0, "doc1", np.zeros(3), 0.5, prov)
        r2 = SearchResult(1, "doc2", np.zeros(3), 0.8, prov)
        sorted_results = sorted([r1, r2])
        assert sorted_results[0].score == 0.8  # Higher first
