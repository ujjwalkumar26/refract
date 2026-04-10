"""Optional embedding providers.

Embedders are optional extras. Install the relevant extra to use them:

- ``pip install 'refract-search[sentence-transformers]'``
- ``pip install 'refract-search[openai]'``
- ``pip install 'refract-search[cohere]'``

The base class is always available for implementing custom embedders.
"""

from refract.embedders.base import BaseEmbedder

__all__ = ["BaseEmbedder"]

# Embedder implementations are NOT imported here to avoid
# importing optional dependencies at package load time.
# Users import them directly:
#   from refract.embedders import SentenceTransformerEmbedder
