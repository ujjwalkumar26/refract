"""Train a learned router from relevance feedback and use it for search.

Run with: python examples/train_learned_router.py
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import refract
from refract.routing import LearnedRouter


def load_sample_dataset() -> tuple[list[str], list[str], dict[int, set[int]]]:
    sample_path = Path(__file__).parent.parent / "samples" / "mini_corpus.json"
    with open(sample_path) as f:
        payload = json.load(f)

    corpus = payload["documents"]
    queries = [item["query"] for item in payload["sample_queries"]]
    relevance = {
        idx: set(item["expected_relevant"]) for idx, item in enumerate(payload["sample_queries"])
    }
    return corpus, queries, relevance


if __name__ == "__main__":
    docs, queries, relevance = load_sample_dataset()

    router = LearnedRouter(["cosine", "bm25", "mahalanobis", "euclidean"])
    evaluation = router.fit_from_relevance(queries, docs, relevance, top_k=5)

    print("=" * 72)
    print("Learned Router Training")
    print("=" * 72)
    print(evaluation)
    print("Per-metric quality:", evaluation.metric_quality)
    print()

    with TemporaryDirectory() as tmp_dir:
        router_path = Path(tmp_dir) / "learned_router.pkl"
        router.save(str(router_path))
        loaded_router = LearnedRouter.load(str(router_path))

        results = refract.search(
            "how can I sort a list in Python",
            docs,
            router=loaded_router,
            top_k=3,
        )

        print("Top results with the trained router:")
        for result in results:
            print(f"  {result.score:.3f}  {result.text}")
