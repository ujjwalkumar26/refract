"""Tests for heuristic router."""

from __future__ import annotations

import numpy as np

from refract.routing.heuristic import HeuristicRouter
from refract.types import QueryProfile, SpaceProfile


def _make_query_profile(
    query_type: str = "natural_language",
    entropy: float = 1.5,
) -> QueryProfile:
    return QueryProfile(
        raw="test query",
        vector=np.zeros(64),
        query_type=query_type,  # type: ignore[arg-type]
        token_count=2,
        embedding_norm=1.0,
        entropy=entropy,
    )


def _make_space_profile(
    density: str = "medium",
    score_spread: float = 0.1,
) -> SpaceProfile:
    return SpaceProfile(
        n_candidates=100,
        embedding_dim=64,
        variance=0.1,
        anisotropy=5.0,
        density=density,  # type: ignore[arg-type]
        score_spread=score_spread,
    )


class TestHeuristicRouter:
    def setup_method(self) -> None:
        self.router = HeuristicRouter()
        self.all_metrics = ["cosine", "bm25", "mahalanobis", "euclidean"]

    def test_weights_sum_to_one(self) -> None:
        for qt in ["keyword", "natural_language", "code", "structured"]:
            for density in ["sparse", "medium", "dense"]:
                qp = _make_query_profile(query_type=qt)
                sp = _make_space_profile(density=density)
                weights = self.router.route(qp, sp, self.all_metrics)
                total = sum(weights.values())
                assert abs(total - 1.0) < 1e-6, f"Failed for {qt}/{density}: sum={total}"

    def test_keyword_has_bm25(self) -> None:
        qp = _make_query_profile(query_type="keyword")
        sp = _make_space_profile(density="sparse")
        weights = self.router.route(qp, sp, self.all_metrics)
        assert weights.get("bm25", 0) > 0

    def test_code_has_bm25(self) -> None:
        qp = _make_query_profile(query_type="code")
        sp = _make_space_profile()
        weights = self.router.route(qp, sp, self.all_metrics)
        assert weights.get("bm25", 0) > 0

    def test_dense_boosts_mahalanobis(self) -> None:
        qp = _make_query_profile(query_type="natural_language")
        sp_sparse = _make_space_profile(density="sparse")
        sp_dense = _make_space_profile(density="dense")
        w_sparse = self.router.route(qp, sp_sparse, self.all_metrics)
        w_dense = self.router.route(qp, sp_dense, self.all_metrics)
        assert w_dense.get("mahalanobis", 0) > w_sparse.get("mahalanobis", 0)

    def test_high_entropy_reduces_dominant(self) -> None:
        qp_low = _make_query_profile(entropy=0.5)
        qp_high = _make_query_profile(entropy=5.0)
        sp = _make_space_profile()
        w_low = self.router.route(qp_low, sp, self.all_metrics)
        w_high = self.router.route(qp_high, sp, self.all_metrics)
        # The dominant metric should have lower weight with high entropy
        dominant = max(w_low, key=w_low.get)  # type: ignore[arg-type]
        assert w_high[dominant] <= w_low[dominant]

    def test_low_spread_boosts_mahalanobis(self) -> None:
        qp = _make_query_profile()
        sp_normal = _make_space_profile(score_spread=0.1)
        sp_low = _make_space_profile(score_spread=0.001)
        w_normal = self.router.route(qp, sp_normal, self.all_metrics)
        w_low = self.router.route(qp, sp_low, self.all_metrics)
        assert w_low.get("mahalanobis", 0) >= w_normal.get("mahalanobis", 0)

    def test_subset_metrics(self) -> None:
        qp = _make_query_profile()
        sp = _make_space_profile()
        weights = self.router.route(qp, sp, ["cosine", "bm25"])
        assert set(weights.keys()) == {"cosine", "bm25"}
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_all_weights_non_negative(self) -> None:
        qp = _make_query_profile(entropy=10.0)  # extreme entropy
        sp = _make_space_profile(score_spread=0.001)  # extreme low spread
        weights = self.router.route(qp, sp, self.all_metrics)
        assert all(w >= 0 for w in weights.values())
