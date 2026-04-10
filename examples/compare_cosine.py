"""Compare refract vs vanilla cosine -- side by side.

Demonstrates that refract's dynamic metric routing produces
different (and often better) rankings than static cosine similarity.

Run with: python examples/compare_cosine.py
"""

import json
from pathlib import Path

import numpy as np

import refract
from refract.metrics.cosine import CosineMetric


def cosine_only_search(query: str, docs: list[str], top_k: int = 5) -> list[tuple[float, str]]:
    """Vanilla cosine-similarity search using TF-IDF vectors."""
    from refract.search import _build_tfidf_vectors

    all_texts = docs + [query]
    all_vecs = _build_tfidf_vectors(all_texts)
    corpus_vecs = all_vecs[:-1]
    query_vec = all_vecs[-1]

    metric = CosineMetric()
    scores = metric.batch_score(query_vec, corpus_vecs)
    indices = np.argsort(-scores)[:top_k]

    return [(float(scores[i]), docs[i]) for i in indices]


if __name__ == "__main__":
    # Load the sample corpus
    corpus_path = Path(__file__).parent.parent / "samples" / "mini_corpus.json"
    with open(corpus_path) as f:
        data = json.load(f)

    docs = data["documents"]
    queries = [q["query"] for q in data["sample_queries"]]

    print("=" * 70)
    print("refract vs Vanilla Cosine -- Side-by-Side Comparison")
    print("=" * 70)

    for query in queries:
        print(f"\n[SEARCH] Query: {query!r}")
        print("-" * 60)

        # Cosine only
        cosine_results = cosine_only_search(query, docs, top_k=3)

        # refract
        refract_results = refract.search(query, docs, top_k=3)

        print(f"\n  {'Cosine Only':40s}  {'refract (dynamic)':40s}")
        print(f"  {'-' * 40}  {'-' * 40}")

        for i in range(3):
            c_score, c_text = cosine_results[i]
            r_result = refract_results[i]
            c_preview = c_text[:35] + "..." if len(c_text) > 35 else c_text
            r_preview = (
                (r_result.text or "")[:35] + "..."
                if len(r_result.text or "") > 35
                else r_result.text
            )
            print(f"  {c_score:.3f} {c_preview:35s}  {r_result.score:.3f} {r_preview}")

        # Show metric breakdown for refract's top result
        prov = refract_results[0].provenance
        print(f"\n  refract breakdown (top result):")
        print(f"    Query type: {prov.query_type}, Density: {prov.space_density}")
        for ms in prov.metric_scores:
            print(
                f"    {ms.metric_name:12s}: score={ms.raw_score:.3f} x weight={ms.weight:.2f} = {ms.weighted_score:.3f}"
            )

    print("\n" + "=" * 70)
    print("Note: refract dynamically weights metrics based on query type and space geometry.")
    print("This is why the rankings may differ -- and often improve -- vs static cosine.")
