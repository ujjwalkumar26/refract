"""Cohere embedder (optional dependency).

Install with: pip install 'refract-search[cohere]'
"""

from __future__ import annotations

import os

import numpy as np

from refract.embedders.base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    """Embedder using Cohere's embedding API.

    Automatically uses appropriate input_type for queries vs documents.

    Example:
        >>> embedder = CohereEmbedder(model="embed-english-v3.0")
        >>> vectors = embedder.embed(["hello world"])
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: str | None = None,
    ) -> None:
        """Initialize with a Cohere model.

        Args:
            model: Cohere embedding model name.
            api_key: Cohere API key. Falls back to COHERE_API_KEY env var.
        """
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "CohereEmbedder requires the cohere package. "
                "Install with:\n  pip install 'refract-search[cohere]'"
            ) from None

        self.model = model
        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise ValueError(
                "Cohere API key required. Pass api_key= or set COHERE_API_KEY env var."
            )
        self._client = cohere.Client(api_key=key)

    def embed(
        self,
        texts: list[str],
        input_type: str = "search_document",
    ) -> np.ndarray:
        """Embed texts using Cohere's API.

        Args:
            texts: List of text strings to embed.
            input_type: "search_document" for corpus, "search_query" for queries.

        Returns:
            Array of shape ``(len(texts), dim)`` with embeddings.
        """
        response = self._client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
        )
        vectors = np.array(response.embeddings, dtype=np.float64)

        # L2 normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        vectors = vectors / norms

        return vectors  # type: ignore[no-any-return]

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single query text.

        Uses "search_query" input type for queries.
        """
        return self.embed([text], input_type="search_query")[0]

    def __repr__(self) -> str:
        return f"CohereEmbedder(model={self.model!r})"
