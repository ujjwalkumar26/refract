"""refract — Context-aware similarity search.

Smart similarity search that understands your query and your data.
Replaces static cosine similarity with dynamic, context-aware metric
routing — weighted based on query type and search space geometry.

Quick start::

    import refract

    docs = [
        "Sort a Python list using the sorted() built-in.",
        "Neural networks learn representations of data.",
        "Retrieve relevant documents from a large corpus.",
    ]

    results = refract.search("how do I sort things in Python", docs)

    for r in results:
        print(f"{r.score:.3f}  {r.text}")

See https://github.com/ujjwalkumar26/refract for full documentation.
"""

from refract._version import __version__
from refract.benchmark.harness import BenchmarkHarness, BenchmarkResult
from refract.embedders.base import BaseEmbedder
from refract.metrics.base import BaseMetric
from refract.metrics.bm25 import BM25Metric
from refract.metrics.cosine import CosineMetric
from refract.metrics.euclidean import EuclideanMetric
from refract.metrics.mahalanobis import MahalanobisMetric
from refract.metrics.registry import DEFAULT_REGISTRY, MetricRegistry
from refract.routing.base import BaseRouter
from refract.routing.composite import CompositeRouter
from refract.routing.heuristic import HeuristicRouter
from refract.routing.learned import LearnedRouter, LearnedRouterEvaluation
from refract.search import search, search_batch
from refract.types import (
    MetricScore,
    Provenance,
    QueryProfile,
    SearchResult,
    SpaceProfile,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "BM25Metric",
    # Embedders
    "BaseEmbedder",
    # Metrics
    "BaseMetric",
    # Routing
    "BaseRouter",
    # Benchmark
    "BenchmarkHarness",
    "BenchmarkResult",
    "CompositeRouter",
    "CosineMetric",
    "EuclideanMetric",
    "HeuristicRouter",
    "LearnedRouter",
    "LearnedRouterEvaluation",
    "MahalanobisMetric",
    "MetricRegistry",
    "MetricScore",
    "Provenance",
    "QueryProfile",
    # Types
    "SearchResult",
    "SpaceProfile",
    # Version
    "__version__",
    # Main API
    "search",
    "search_batch",
]
