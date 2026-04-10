"""Vector DB integration example -- using refract with pre-computed vectors.

Shows how refract integrates with vector databases like FAISS, Qdrant,
or Pinecone. You retrieve candidate vectors from your DB, then use
refract for smarter scoring.

Run with: python examples/vector_db_integration.py
"""

import numpy as np

import refract

# ── Simulated vector database ──────────────────────────────────────────────


class MockVectorDB:
    """Simulates a vector database that stores and retrieves embeddings.

    In production, this would be FAISS, Qdrant, Pinecone, Weaviate, etc.
    """

    def __init__(self, vectors: np.ndarray, texts: list[str]) -> None:
        self.vectors = vectors
        self.texts = texts

    def approximate_search(self, query_vec: np.ndarray, top_k: int = 50) -> tuple:
        """Simulate ANN search -- returns candidate vectors and texts.

        In production, this would use HNSW, IVF, etc. for sublinear search.
        """
        # Simple exhaustive search as a stand-in for ANN
        from refract.metrics.cosine import CosineMetric

        scores = CosineMetric().batch_score(query_vec, self.vectors)
        top_indices = np.argsort(-scores)[:top_k]
        return self.vectors[top_indices], [self.texts[i] for i in top_indices]


# ── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate a large corpus with embeddings
    n_docs = 1000
    dim = 128
    corpus_vectors = rng.standard_normal((n_docs, dim)).astype(np.float64)
    # L2 normalize
    norms = np.linalg.norm(corpus_vectors, axis=1, keepdims=True)
    corpus_vectors = corpus_vectors / norms

    corpus_texts = [
        f"Document {i}: {rng.choice(['ML', 'search', 'NLP', 'code', 'data'])} topic"
        for i in range(n_docs)
    ]

    # Create "vector database"
    db = MockVectorDB(corpus_vectors, corpus_texts)

    # Query
    query_vec = rng.standard_normal(dim).astype(np.float64)
    query_vec = query_vec / np.linalg.norm(query_vec)

    print("=" * 60)
    print("refract with Vector DB Integration")
    print("=" * 60)

    # Step 1: ANN retrieval from vector DB (fast, approximate)
    candidate_vecs, candidate_texts = db.approximate_search(query_vec, top_k=50)
    print(f"\n[DB] Retrieved {len(candidate_vecs)} candidates from vector DB")

    # Step 2: refract re-scoring (precise, context-aware)
    results = refract.search(query_vec, candidate_vecs, top_k=10)

    print(f"[SEARCH] Re-scored with refract (top 10):\n")
    for r in results:
        print(f"  Score: {r.score:.4f}  |  Candidate #{r.index}")
        breakdown = ", ".join(
            f"{ms.metric_name}={ms.raw_score:.3f}x{ms.weight:.2f}"
            for ms in r.provenance.metric_scores
        )
        print(f"         -> {breakdown}")

    print(f"\n[OK] Vector DB provides candidates (speed)")
    print(f"[OK] refract provides smarter scoring (quality)")
    print(f"[OK] Together: fast AND accurate retrieval")
