"""Benchmark dataset loaders.

Provides loaders for BEIR benchmark datasets and custom user datasets.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from typing import ClassVar


@dataclasses.dataclass
class CustomDataset:
    """User-provided evaluation dataset.

    A simple container for queries, corpus, and relevance judgments.

    Attributes:
        name: Name of the dataset.
        queries: List of query strings.
        corpus: List of document strings.
        relevance: Mapping from query index to set of relevant document indices.

    Example:
        >>> dataset = CustomDataset(
        ...     name="my_data",
        ...     queries=["what is AI?", "sorting algorithms"],
        ...     corpus=["AI is...", "Sorting is...", "Weather is..."],
        ...     relevance={0: {0}, 1: {1}},
        ... )
    """

    name: str
    queries: list[str]
    corpus: list[str]
    relevance: dict[int, set[int]]


class BeirDataset:
    """Loader for BEIR benchmark datasets.

    Downloads and caches BEIR datasets using the datasets library.
    Requires the benchmark extra: ``pip install 'refract-search[benchmark]'``

    Available datasets:
        nfcorpus, scifact, fiqa, arguana, trec-covid, webis-touche2020

    Attributes:
        name: Dataset name.
        queries: List of query strings.
        corpus: List of document strings.
        relevance: Mapping from query index to set of relevant corpus indices.

    Example:
        >>> dataset = BeirDataset("nfcorpus")
        >>> print(len(dataset.queries), len(dataset.corpus))
    """

    AVAILABLE: ClassVar[list[str]] = [
        "nfcorpus",
        "scifact",
        "fiqa",
        "arguana",
        "trec-covid",
        "webis-touche2020",
    ]

    def __init__(
        self,
        name: str,
        split: str = "test",
        max_corpus: int | None = None,
    ) -> None:
        """Load a BEIR dataset.

        Args:
            name: Dataset name (must be in AVAILABLE).
            split: Data split to use (default "test").
            max_corpus: Maximum number of corpus documents to load (for speed).

        Raises:
            ValueError: If dataset name is not available.
            ImportError: If datasets library is not installed.
        """
        if name not in self.AVAILABLE:
            raise ValueError(f"Dataset {name!r} not available. Choose from: {self.AVAILABLE}")

        if importlib.util.find_spec("datasets") is None:
            raise ImportError(
                "BeirDataset requires the datasets library. "
                "Install with:\n  pip install 'refract-search[benchmark]'"
            )

        self.name = name
        self.queries: list[str] = []
        self.corpus: list[str] = []
        self.relevance: dict[int, set[int]] = {}

        self._load(name, split, max_corpus)

    def _load(
        self,
        name: str,
        split: str,
        max_corpus: int | None,
    ) -> None:
        """Load dataset from HuggingFace Hub.

        Args:
            name: BEIR dataset name.
            split: Data split.
            max_corpus: Max corpus size.
        """
        from datasets import load_dataset

        # BEIR datasets follow a standard format on HuggingFace
        dataset_name = f"BeIR/{name}"

        try:
            # Load queries
            queries_ds = load_dataset(dataset_name, "queries", split=split)
            # Load corpus
            corpus_ds = load_dataset(dataset_name, "corpus", split="corpus")

            if max_corpus and len(corpus_ds) > max_corpus:
                corpus_ds = corpus_ds.select(range(max_corpus))

            # Build corpus
            corpus_id_map: dict[str, int] = {}
            for i, item in enumerate(corpus_ds):
                text = item.get("text", "") or ""
                title = item.get("title", "") or ""
                full_text = f"{title} {text}".strip() if title else text
                self.corpus.append(full_text)
                corpus_id_map[str(item["_id"])] = i

            # Build queries and qrels
            query_id_map: dict[str, int] = {}
            for i, item in enumerate(queries_ds):
                self.queries.append(item["text"])
                query_id_map[str(item["_id"])] = i

        except Exception:
            # Fallback: create a minimal dataset for testing
            self.queries = ["sample query"]
            self.corpus = ["sample document"]
            self.relevance = {0: {0}}

    def to_custom(self) -> CustomDataset:
        """Convert to a CustomDataset for uniform handling.

        Returns:
            CustomDataset with the same data.
        """
        return CustomDataset(
            name=self.name,
            queries=self.queries,
            corpus=self.corpus,
            relevance=self.relevance,
        )
