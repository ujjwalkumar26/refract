"""Built-in similarity metrics.

Provides cosine, euclidean, mahalanobis, and BM25 metrics, plus
the BaseMetric interface for custom metrics and a MetricRegistry
for discovery.
"""

from refract.metrics.base import BaseMetric
from refract.metrics.bm25 import BM25Metric
from refract.metrics.cosine import CosineMetric
from refract.metrics.euclidean import EuclideanMetric
from refract.metrics.mahalanobis import MahalanobisMetric
from refract.metrics.registry import DEFAULT_REGISTRY, MetricRegistry

__all__ = [
    "DEFAULT_REGISTRY",
    "BM25Metric",
    "BaseMetric",
    "CosineMetric",
    "EuclideanMetric",
    "MahalanobisMetric",
    "MetricRegistry",
]
