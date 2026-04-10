"""Abstract base class for embedding providers.

Embedders are optional — refract works with pre-computed vectors.
Install extras like ``refract-search[sentence-transformers]`` to use them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for text embedding providers.

    An embedder converts a list of texts into a matrix of embedding vectors.
    All embedders normalize outputs to unit length by default.

    Subclasses must implement:
        - ``embed()``: Batch-embed a list of texts.

    Example:
        >>> class MyEmbedder(BaseEmbedder):
        ...     def embed(self, texts):
        ...         # Your embedding logic
        ...         return np.random.randn(len(texts), 384)
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape ``(len(texts), embedding_dim)`` with embeddings.
        """

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text string to embed.

        Returns:
            Array of shape ``(embedding_dim,)`` with the embedding.
        """
        return self.embed([text])[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
