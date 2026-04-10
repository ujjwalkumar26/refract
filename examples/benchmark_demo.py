"""Benchmark demo — evaluate refract on a custom dataset.

Shows how to use the BenchmarkHarness to measure search quality
with standard IR metrics (NDCG, Recall, MRR).

Run with: python examples/benchmark_demo.py
"""

import json
from pathlib import Path

from refract.benchmark import BenchmarkHarness, CustomDataset


if __name__ == "__main__":
    # Load sample data
    corpus_path = Path(__file__).parent.parent / "samples" / "mini_corpus.json"
    with open(corpus_path) as f:
        data = json.load(f)

    docs = data["documents"]
    sample_queries = data["sample_queries"]

    # Build a CustomDataset with relevance judgments
    queries = [q["query"] for q in sample_queries]
    relevance = {
        i: set(q["expected_relevant"])
        for i, q in enumerate(sample_queries)
    }

    dataset = CustomDataset(
        name="mini_corpus",
        queries=queries,
        corpus=docs,
        relevance=relevance,
    )

    # Run benchmark
    harness = BenchmarkHarness()
    results = harness.run(
        dataset=dataset,
        compare_cosine_baseline=True,
    )

    # Display results
    print("=" * 70)
    print("refract Benchmark Results")
    print("=" * 70)
    print()
    print(f"{'Method':<20s} {'NDCG@10':>8s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s} {'Latency':>10s}")
    print("-" * 70)

    for r in results:
        print(
            f"{r.method:<20s} "
            f"{r.ndcg_at_10:>8.3f} "
            f"{r.recall_at_1:>6.3f} "
            f"{r.recall_at_5:>6.3f} "
            f"{r.recall_at_10:>6.3f} "
            f"{r.mrr_score:>6.3f} "
            f"{r.avg_latency_ms:>8.1f}ms"
        )

    print()
    print("Higher NDCG/Recall/MRR = better retrieval quality.")
