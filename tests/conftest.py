"""Shared test fixtures for refract tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_corpus_texts() -> list[str]:
    """A small text corpus for testing."""
    return [
        "Sort a Python list using the sorted() built-in function.",
        "Neural networks learn hierarchical representations of data.",
        "Retrieve relevant documents from a large corpus efficiently.",
        "Use cosine similarity to measure vector closeness in embedding space.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require training data and compute.",
        "Python decorators modify function behavior at definition time.",
        "Information retrieval systems rank documents by relevance.",
        "Deep learning architectures include CNNs, RNNs, and Transformers.",
        "Binary search has O(log n) time complexity.",
    ]


@pytest.fixture
def sample_vectors() -> np.ndarray:
    """Deterministic sample vectors for testing (10 x 64)."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((10, 64))
    # L2 normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def sample_query_vec() -> np.ndarray:
    """A single deterministic query vector (64,)."""
    rng = np.random.default_rng(123)
    vec = rng.standard_normal(64)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def sample_query_text() -> str:
    """A sample query string."""
    return "how do I sort things in Python"


@pytest.fixture
def zero_vector() -> np.ndarray:
    """A zero vector for edge case testing."""
    return np.zeros(64)


@pytest.fixture
def identical_vectors() -> np.ndarray:
    """Corpus where all vectors are identical (degenerate case)."""
    vec = np.ones(64) / np.sqrt(64)
    return np.tile(vec, (10, 1))
