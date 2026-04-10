"""Tests for query analyzer."""

from __future__ import annotations

import numpy as np

from refract.analysis.query_analyzer import _compute_entropy, _detect_query_type, analyze_query


class TestDetectQueryType:
    def test_code_python_function(self) -> None:
        assert _detect_query_type("def foo(x): return x + 1") == "code"

    def test_code_class(self) -> None:
        assert _detect_query_type("class MyModel(nn.Module):") == "code"

    def test_code_import(self) -> None:
        assert _detect_query_type("import numpy as np") == "code"

    def test_code_arrow(self) -> None:
        assert _detect_query_type("def foo(x) -> int:") == "code"

    def test_keyword_single(self) -> None:
        assert _detect_query_type("python") == "keyword"

    def test_keyword_short(self) -> None:
        assert _detect_query_type("sort list python") == "keyword"

    def test_keyword_four_words(self) -> None:
        assert _detect_query_type("best sorting algorithm python") == "keyword"

    def test_natural_language(self) -> None:
        result = _detect_query_type("How do I sort a list in Python?")
        assert result == "natural_language"

    def test_natural_language_long(self) -> None:
        result = _detect_query_type("what are the philosophical implications of determinism")
        assert result == "natural_language"

    def test_empty_string(self) -> None:
        assert _detect_query_type("") == "keyword"

    def test_structured_json(self) -> None:
        assert _detect_query_type('{"name": "test"}') == "structured"


class TestComputeEntropy:
    def test_uniform_scores(self) -> None:
        scores = np.ones(10) * 0.5
        entropy = _compute_entropy(scores)
        # Uniform → max entropy
        assert entropy > 2.0

    def test_dominant_score(self) -> None:
        scores = np.zeros(10)
        scores[0] = 100.0
        entropy = _compute_entropy(scores)
        # One dominant → low entropy
        assert entropy < 0.5

    def test_empty(self) -> None:
        assert _compute_entropy(np.array([])) == 0.0

    def test_single(self) -> None:
        assert _compute_entropy(np.array([1.0])) == 0.0


class TestAnalyzeQuery:
    def test_text_only(self) -> None:
        qp = analyze_query(query_text="test query", query_vector=None)
        assert qp.query_type == "keyword"
        assert qp.token_count == 2
        assert qp.embedding_norm == 0.0

    def test_vector_only(self) -> None:
        vec = np.array([1.0, 2.0, 3.0])
        qp = analyze_query(query_text=None, query_vector=vec)
        assert qp.query_type == "natural_language"
        assert qp.embedding_norm > 0.0

    def test_with_candidates(
        self, sample_query_vec: np.ndarray, sample_vectors: np.ndarray
    ) -> None:
        qp = analyze_query(
            query_text="how to sort a list",
            query_vector=sample_query_vec,
            candidate_vectors=sample_vectors,
        )
        assert qp.entropy > 0.0
        assert qp.query_type == "natural_language"
