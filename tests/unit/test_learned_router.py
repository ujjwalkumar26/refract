"""Tests for the learned router."""

from __future__ import annotations

import numpy as np

import refract
from refract.routing.learned import LearnedRouter
from refract.types import QueryProfile, SpaceProfile


def _query_profile(query_type: str, token_count: int, entropy: float) -> QueryProfile:
    return QueryProfile(
        raw="sample query",
        vector=np.ones(8, dtype=np.float64),
        query_type=query_type,  # type: ignore[arg-type]
        token_count=token_count,
        embedding_norm=float(np.sqrt(8.0)),
        entropy=entropy,
    )


def _space_profile(density: str, variance: float, score_spread: float) -> SpaceProfile:
    return SpaceProfile(
        n_candidates=32,
        embedding_dim=8,
        variance=variance,
        anisotropy=3.5,
        density=density,  # type: ignore[arg-type]
        score_spread=score_spread,
    )


class TestLearnedRouter:
    def test_fit_route_and_save_load(self, tmp_path) -> None:
        router = LearnedRouter(["cosine", "bm25"])
        query_profiles = [
            _query_profile("keyword", 2, 0.6),
            _query_profile("natural_language", 6, 2.4),
            _query_profile("keyword", 3, 0.8),
            _query_profile("natural_language", 8, 2.7),
        ]
        space_profiles = [
            _space_profile("sparse", 0.25, 0.20),
            _space_profile("dense", 0.03, 0.01),
            _space_profile("sparse", 0.22, 0.18),
            _space_profile("dense", 0.04, 0.02),
        ]
        targets = [
            {"cosine": 0.1, "bm25": 0.9},
            {"cosine": 0.85, "bm25": 0.15},
            {"cosine": 0.2, "bm25": 0.8},
            {"cosine": 0.9, "bm25": 0.1},
        ]

        router.fit(query_profiles, space_profiles, targets)
        keyword_weights = router.route(query_profiles[0], space_profiles[0], ["cosine", "bm25"])
        natural_weights = router.route(query_profiles[1], space_profiles[1], ["cosine", "bm25"])

        assert abs(sum(keyword_weights.values()) - 1.0) < 1e-6
        assert keyword_weights["bm25"] > keyword_weights["cosine"]
        assert natural_weights["cosine"] > natural_weights["bm25"]

        path = tmp_path / "router.pkl"
        router.save(str(path))
        restored = LearnedRouter.load(str(path))
        restored_weights = restored.route(query_profiles[0], space_profiles[0], ["cosine", "bm25"])
        assert restored_weights["bm25"] > restored_weights["cosine"]

    def test_fit_from_relevance_and_use_in_search(self) -> None:
        docs = [
            "Sort a Python list with the sorted built-in.",
            "Python list comprehensions create lists concisely.",
            "Neural networks learn with backpropagation.",
            "Transformer architectures power modern deep learning.",
            "JSON records store structured key value pairs.",
            "Tabular data can be queried with structured filters.",
        ]
        queries = [
            "python sort list",
            "neural network training",
            '{"city": "Delhi"}',
        ]
        relevance = {
            0: {0, 1},
            1: {2, 3},
            2: {4, 5},
        }

        router = LearnedRouter(["cosine", "bm25", "euclidean"])
        evaluation = router.fit_from_relevance(queries, docs, relevance, top_k=3)

        assert evaluation.n_queries == 3
        assert 0.0 <= evaluation.weight_mae <= 1.0
        assert 0.0 <= evaluation.router_ndcg_at_k <= 1.0
        assert set(evaluation.metric_quality) == {"cosine", "bm25", "euclidean"}

        results = refract.search("sort a list in python", docs, router=router, top_k=2)
        assert results[0].text is not None
        assert "Python" in results[0].text
        assert results[0].provenance.router_name == "learned"
