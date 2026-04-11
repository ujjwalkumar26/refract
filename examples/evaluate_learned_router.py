"""Evaluate heuristic vs learned routing on a simple train/test split.

Run with: python examples/evaluate_learned_router.py
"""

from __future__ import annotations

import json
from pathlib import Path

from refract.benchmark import BenchmarkHarness, CustomDataset
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
    corpus, queries, relevance = load_sample_dataset()
    train_queries = queries[:3]
    train_relevance = {idx: relevance[idx] for idx in range(3)}

    test_queries = queries[3:]
    test_relevance = {idx: relevance[idx + 3] for idx in range(len(test_queries))}

    router = LearnedRouter(["cosine", "bm25", "mahalanobis", "euclidean"])
    training_report = router.fit_from_relevance(train_queries, corpus, train_relevance, top_k=5)

    test_dataset = CustomDataset(
        name="mini_corpus_test_split",
        queries=test_queries,
        corpus=corpus,
        relevance=test_relevance,
    )

    harness = BenchmarkHarness()
    heuristic_result = harness.run(test_dataset, compare_cosine_baseline=False)[0]
    learned_result = harness.run(
        test_dataset,
        router=router,
        compare_cosine_baseline=False,
    )[0]

    print("=" * 72)
    print("Training Report")
    print("=" * 72)
    print(training_report)
    print()
    print("=" * 72)
    print("Held-Out Evaluation")
    print("=" * 72)
    print(f"{'Router':<12s} {'NDCG@10':>8s} {'Recall@10':>10s} {'MRR':>8s}")
    print("-" * 72)
    print(
        f"{'heuristic':<12s} "
        f"{heuristic_result.ndcg_at_10:>8.3f} "
        f"{heuristic_result.recall_at_10:>10.3f} "
        f"{heuristic_result.mrr_score:>8.3f}"
    )
    print(
        f"{'learned':<12s} "
        f"{learned_result.ndcg_at_10:>8.3f} "
        f"{learned_result.recall_at_10:>10.3f} "
        f"{learned_result.mrr_score:>8.3f}"
    )
