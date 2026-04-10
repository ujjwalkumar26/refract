"""OpenAI embedder (optional dependency).

Install with: pip install 'refract-search[openai]'
"""

from __future__ import annotations

import os

import numpy as np

from refract.embedders.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's embedding API.

    Supports batching (up to 100 texts per API call) and automatic
    L2 normalization.

    Example:
        >>> embedder = OpenAIEmbedder(model="text-embedding-3-small")
        >>> vectors = embedder.embed(["hello world"])
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        """Initialize with an OpenAI model.

        Args:
            model: OpenAI embedding model name.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAIEmbedder requires the openai package. "
                "Install with:\n  pip install 'refract-search[openai]'"
            ) from None

        self.model = model
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Pass api_key= or set OPENAI_API_KEY env var."
            )
        self._client = openai.OpenAI(api_key=key)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using OpenAI's API.

        Batches requests in chunks of 100 to respect API limits.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape ``(len(texts), dim)`` with L2-normalized embeddings.
        """
        all_embeddings: list[list[float]] = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(input=batch, model=self.model)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        vectors = np.array(all_embeddings, dtype=np.float64)

        # L2 normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        vectors = vectors / norms

        return vectors

    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model={self.model!r})"
