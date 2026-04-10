"""Tests for all similarity metrics."""

from __future__ import annotations

import numpy as np
import pytest

from refract.metrics.bm25 import BM25Metric
from refract.metrics.cosine import CosineMetric
from refract.metrics.euclidean import EuclideanMetric
from refract.metrics.mahalanobis import MahalanobisMetric
from refract.metrics.registry import MetricRegistry, create_default_registry


class TestCosineMetric:
    def setup_method(self) -> None:
        self.metric = CosineMetric()

    def test_identical_vectors(self) -> None:
        vec = np.array([1.0, 2.0, 3.0])
        score = self.metric.score(vec, vec)
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        score = self.metric.score(v1, v2)
        assert abs(score - 0.5) < 1e-6  # cos=0, mapped to 0.5

    def test_opposite_vectors(self) -> None:
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        score = self.metric.score(v1, v2)
        assert abs(score - 0.0) < 1e-6  # cos=-1, mapped to 0

    def test_zero_vector(self, zero_vector: np.ndarray) -> None:
        vec = np.ones(64)
        score = self.metric.score(zero_vector, vec)
        assert score == 0.0

    def test_batch_score_shape(self, sample_query_vec: np.ndarray, sample_vectors: np.ndarray) -> None:
        scores = self.metric.batch_score(sample_query_vec, sample_vectors)
        assert scores.shape == (10,)

    def test_batch_matches_single(self, sample_query_vec: np.ndarray, sample_vectors: np.ndarray) -> None:
        batch = self.metric.batch_score(sample_query_vec, sample_vectors)
        for i in range(len(sample_vectors)):
            single = self.metric.score(sample_query_vec, sample_vectors[i])
            assert abs(batch[i] - single) < 1e-6

    def test_score_range(self, sample_query_vec: np.ndarray, sample_vectors: np.ndarray) -> None:
        scores = self.metric.batch_score(sample_query_vec, sample_vectors)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


class TestEuclideanMetric:
    def setup_method(self) -> None:
        self.metric = EuclideanMetric()

    def test_identical_vectors(self) -> None:
        vec = np.array([1.0, 2.0, 3.0])
        score = self.metric.score(vec, vec)
        assert abs(score - 1.0) < 1e-6

    def test_distant_vectors(self) -> None:
        v1 = np.zeros(3)
        v2 = np.array([100.0, 100.0, 100.0])
        score = self.metric.score(v1, v2)
        assert score < 0.01

    def test_batch_score_shape(self, sample_query_vec: np.ndarray, sample_vectors: np.ndarray) -> None:
        scores = self.metric.batch_score(sample_query_vec, sample_vectors)
        assert scores.shape == (10,)

    def test_score_range(self, sample_query_vec: np.ndarray, sample_vectors: np.ndarray) -> None:
        scores = self.metric.batch_score(sample_query_vec, sample_vectors)
        assert np.all(scores > 0.0)
        assert np.all(scores <= 1.0)


class TestMahalanobisMetric:
    def test_requires_fitting(self, sample_query_vec: np.ndarray) -> None:
        metric = MahalanobisMetric()
        with pytest.raises(RuntimeError, match="fit"):
            metric.score(sample_query_vec, sample_query_vec)

    def test_fit_and_score(self, sample_query_vec: np.ndarray, sample_vectors: np.ndarray) -> None:
        metric = MahalanobisMetric()
        metric.fit(sample_vectors)
        score = metric.score(sample_query_vec, sample_vectors[0])
        assert 0.0 < score <= 1.0

    def test_identical_after_fit(self, sample_vectors: np.ndarray) -> None:
        metric = MahalanobisMetric()
        metric.fit(sample_vectors)
        score = metric.score(sample_vectors[0], sample_vectors[0])
        assert abs(score - 1.0) < 1e-6

    def test_batch_score_shape(self, sample_query_vec: np.ndarray, sample_vectors: np.ndarray) -> None:
        metric = MahalanobisMetric()
        metric.fit(sample_vectors)
        scores = metric.batch_score(sample_query_vec, sample_vectors)
        assert scores.shape == (10,)

    def test_too_few_vectors(self) -> None:
        metric = MahalanobisMetric()
        with pytest.raises(ValueError, match="at least 2"):
            metric.fit(np.array([[1.0, 2.0]]))


class TestBM25Metric:
    def test_fit_and_score(self) -> None:
        corpus = [
            "Python is a programming language",
            "Java is also a language",
            "The weather is nice today",
        ]
        metric = BM25Metric(corpus)
        scores = metric.batch_score_text("Python programming")
        assert scores.shape == (3,)
        # Python doc should score highest
        assert scores[0] > scores[2]

    def test_exact_match_highest(self) -> None:
        corpus = ["sort a list", "neural networks", "deep learning"]
        metric = BM25Metric(corpus)
        scores = metric.batch_score_text("sort a list")
        assert np.argmax(scores) == 0

    def test_not_fitted(self) -> None:
        metric = BM25Metric()
        with pytest.raises(RuntimeError, match="fit"):
            metric.batch_score_text("test")

    def test_empty_query(self) -> None:
        metric = BM25Metric(["doc1", "doc2"])
        scores = metric.batch_score_text("!@#$%")
        assert np.all(scores == 0.0)


class TestMetricRegistry:
    def test_default_registry(self) -> None:
        registry = create_default_registry()
        assert "cosine" in registry
        assert "euclidean" in registry
        assert "mahalanobis" in registry
        assert len(registry) == 3

    def test_register_custom(self) -> None:
        registry = MetricRegistry()
        registry.register(CosineMetric())
        assert registry.get("cosine") is not None

    def test_get_or_raise(self) -> None:
        registry = MetricRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get_or_raise("nonexistent")

    def test_list_available(self) -> None:
        registry = create_default_registry()
        names = registry.list_available()
        assert names == ["cosine", "euclidean", "mahalanobis"]
