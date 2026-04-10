"""BM25 sparse lexical similarity metric.

Operates on raw text, not vectors. Uses the rank_bm25 library for
efficient TF-IDF-based scoring. Complements dense vector metrics by
capturing exact keyword matches that embeddings may miss.
"""

from __future__ import annotations

import re

import numpy as np

from refract.metrics.base import BaseMetric


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer.

    Lowercases text, splits on non-alphanumeric characters, and
    filters out tokens shorter than 2 characters.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase token strings.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


class BM25Metric(BaseMetric):
    """BM25 sparse lexical similarity metric.

    Unlike vector-based metrics, BM25 operates on raw text. It captures
    exact keyword matches using TF-IDF weighting, complementing dense
    metrics that may miss surface-level term overlap.

    This metric must be initialized with the corpus text. The fusion
    engine handles this automatically.

    Scores are normalized to [0, 1] by dividing by the maximum score
    across the corpus for a given query.
    """

    name = "bm25"
    is_text_metric = True

    def __init__(
        self,
        corpus_texts: list[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize BM25 metric.

        Args:
            corpus_texts: List of corpus document strings. Can be provided
                later via ``fit_text()``.
            k1: BM25 term frequency saturation parameter.
            b: BM25 length normalization parameter.
        """
        self._k1 = k1
        self._b = b
        self._bm25: object | None = None
        self._corpus_texts: list[str] | None = None
        self._fitted = False

        if corpus_texts is not None:
            self.fit_text(corpus_texts)

    def fit_text(self, corpus_texts: list[str]) -> None:
        """Build the BM25 index from corpus texts.

        Args:
            corpus_texts: List of document strings.
        """
        from rank_bm25 import BM25Okapi

        self._corpus_texts = corpus_texts
        tokenized_corpus = [_tokenize(doc) for doc in corpus_texts]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self._k1, b=self._b)
        self._fitted = True

    def score(self, query_vec: np.ndarray, candidate_vec: np.ndarray) -> float:
        """Not used for BM25 — use batch_score_text instead.

        This exists to satisfy the BaseMetric interface. BM25 scoring
        is handled specially by the fusion engine via ``batch_score_text()``.

        Returns:
            0.0 always. Use batch_score_text for actual BM25 scores.
        """
        return 0.0

    def batch_score_text(self, query_text: str) -> np.ndarray:
        """Compute BM25 scores for query against entire corpus.

        Args:
            query_text: The query string.

        Returns:
            Array of shape ``(n_corpus,)`` with BM25 scores normalized to [0, 1].

        Raises:
            RuntimeError: If the BM25 index has not been built.
        """
        if not self._fitted or self._bm25 is None:
            raise RuntimeError(
                "BM25Metric.fit_text(corpus) must be called before scoring. "
                "This is normally handled automatically by refract.search()."
            )
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return np.zeros(len(self._corpus_texts or []), dtype=np.float64)

        import typing
        scores = typing.cast("typing.Any", self._bm25).get_scores(query_tokens)  # type: ignore
        scores = np.array(scores, dtype=np.float64)

        # Normalize to [0, 1]
        max_score = scores.max()
        if max_score > 1e-12:
            scores = scores / max_score
        return scores  # type: ignore[no-any-return]

    def batch_score(self, query_vec: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Fallback for vector-based interface. Returns zeros.

        BM25 does not use vectors. The fusion engine calls
        ``batch_score_text()`` instead.
        """
        return np.zeros(len(candidates), dtype=np.float64)
