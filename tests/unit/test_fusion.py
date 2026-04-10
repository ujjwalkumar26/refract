"""Tests for fusion engine."""

from __future__ import annotations

import numpy as np

from refract.fusion.weighted import fuse
from refract.metrics.cosine import CosineMetric
from refract.metrics.euclidean import EuclideanMetric
from refract.types import QueryProfile, SpaceProfile


def _make_profiles() -> tuple:
    qp = QueryProfile(
        raw="test",
        vector=np.array([1.0, 0.0, 0.0]),
        query_type="keyword",
        token_count=1,
        embedding_norm=1.0,
        entropy=1.0,
    )
    sp = SpaceProfile(
        n_candidates=3,
        embedding_dim=3,
        variance=0.1,
        anisotropy=2.0,
        density="medium",
        score_spread=0.05,
    )
    return qp, sp


class TestFuse:
    def test_basic_fusion(self) -> None:
        qp, sp = _make_profiles()
        candidates = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ])
        metrics = {
            "cosine": CosineMetric(),
            "euclidean": EuclideanMetric(),
        }
        weights = {"cosine": 0.7, "euclidean": 0.3}

        results = fuse(qp, sp, candidates, None, weights, metrics, "test_router")

        assert len(results) == 3
        # Results should be sorted by score descending
        assert results[0][1] >= results[1][1] >= results[2][1]
        # First result should be the identical vector
        assert results[0][0] == 0

    def test_provenance_structure(self) -> None:
        qp, sp = _make_profiles()
        candidates = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        metrics = {"cosine": CosineMetric()}
        weights = {"cosine": 1.0}

        results = fuse(qp, sp, candidates, None, weights, metrics, "heuristic")

        _, _, prov = results[0]
        assert prov.router_name == "heuristic"
        assert prov.query_type == "keyword"
        assert len(prov.metric_scores) == 1
        assert prov.metric_scores[0].metric_name == "cosine"

    def test_skip_negligible_weights(self) -> None:
        qp, sp = _make_profiles()
        candidates = np.array([[1.0, 0.0, 0.0]])
        metrics = {
            "cosine": CosineMetric(),
            "euclidean": EuclideanMetric(),
        }
        weights = {"cosine": 0.995, "euclidean": 0.005}

        results = fuse(qp, sp, candidates, None, weights, metrics, "test")

        # Euclidean should be skipped (weight < 0.01)
        _, _, prov = results[0]
        metric_names = [ms.metric_name for ms in prov.metric_scores]
        assert "cosine" in metric_names
        # euclidean's weight is below threshold, should be skipped
        assert "euclidean" not in metric_names

    def test_scores_in_valid_range(self) -> None:
        qp, sp = _make_profiles()
        rng = np.random.default_rng(42)
        candidates = rng.standard_normal((20, 3))
        metrics = {"cosine": CosineMetric(), "euclidean": EuclideanMetric()}
        weights = {"cosine": 0.6, "euclidean": 0.4}

        results = fuse(qp, sp, candidates, None, weights, metrics, "test")

        for _, score, _ in results:
            assert 0.0 <= score <= 1.0
