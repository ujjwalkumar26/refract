"""SentenceTransformer embedder (optional dependency).

Install with: pip install 'refract-search[sentence-transformers]'
"""

from __future__ import annotations

import numpy as np

from refract.embedders.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using sentence-transformers models.

    Wraps the sentence-transformers library for convenient embedding.
    Models are cached after first load.

    Example:
        >>> embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        >>> vectors = embedder.embed(["hello world", "foo bar"])
        >>> vectors.shape
        (2, 384)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize with a sentence-transformers model.

        Args:
            model_name: Name of the model (from HuggingFace Hub or local path).
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "SentenceTransformerEmbedder requires sentence-transformers. "
                "Install with:\n  pip install 'refract-search[sentence-transformers]'"
            ) from None

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the sentence-transformers model.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape ``(len(texts), dim)`` with L2-normalized embeddings.
        """
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float64)

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedder(model={self.model_name!r})"
