"""Tests for space analyzer."""

from __future__ import annotations

import numpy as np

from refract.analysis.space_analyzer import _classify_density, _compute_anisotropy, analyze_space


class TestClassifyDensity:
    def test_dense(self) -> None:
        assert _classify_density(0.01) == "dense"

    def test_medium(self) -> None:
        assert _classify_density(0.10) == "medium"

    def test_sparse(self) -> None:
        assert _classify_density(0.30) == "sparse"


class TestComputeAnisotropy:
    def test_isotropic(self) -> None:
        # Isotropic data → anisotropy close to 1
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((100, 10))
        aniso = _compute_anisotropy(vecs)
        assert aniso >= 1.0
        assert aniso < 10.0  # Should be relatively low

    def test_anisotropic(self) -> None:
        # Highly anisotropic: one dimension dominates
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((100, 10))
        vecs[:, 0] *= 100  # Scale first dimension
        aniso = _compute_anisotropy(vecs)
        assert aniso > 10.0  # Should be high

    def test_single_vector(self) -> None:
        vecs = np.array([[1.0, 2.0, 3.0]])
        assert _compute_anisotropy(vecs) == 1.0

    def test_minimum_one(self) -> None:
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((50, 10))
        assert _compute_anisotropy(vecs) >= 1.0


class TestAnalyzeSpace:
    def test_basic(self, sample_vectors: np.ndarray) -> None:
        sp = analyze_space(sample_vectors)
        assert sp.n_candidates == 10
        assert sp.embedding_dim == 64
        assert sp.variance > 0
        assert sp.anisotropy >= 1.0
        assert sp.density in ("sparse", "medium", "dense")

    def test_with_scores(self, sample_vectors: np.ndarray) -> None:
        scores = np.random.default_rng(42).uniform(0, 1, 10)
        sp = analyze_space(sample_vectors, cosine_scores=scores)
        assert sp.score_spread > 0

    def test_without_scores(self, sample_vectors: np.ndarray) -> None:
        sp = analyze_space(sample_vectors)
        assert sp.score_spread == 0.0
