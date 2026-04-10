# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-04-11

### Added

- Core `refract.search()` API with dynamic metric routing
- `refract.search_batch()` for efficient multi-query search
- **Metrics:** Cosine, Euclidean, Mahalanobis, BM25
- **Routing:** `HeuristicRouter` (default), `LearnedRouter`, `CompositeRouter`
- **Analysis:** Automatic query type detection and search space profiling
- **Fusion:** Weighted score fusion with full provenance traces
- **Embedders:** SentenceTransformer, OpenAI, Cohere (optional extras)
- **Benchmark:** Evaluation harness with NDCG, Recall, MRR metrics
- Custom metric and router plugin support via `BaseMetric` / `BaseRouter`
- Score provenance on every result — explainable by default
- Metric registry with register/lookup by name
- `py.typed` marker for PEP 561 type stub support
- Comprehensive test suite (unit + integration)
- Examples: quickstart, RAG pipeline, code search, custom metric, benchmarks

[Unreleased]: https://github.com/ujjwalkumar26/refract/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ujjwalkumar26/refract/releases/tag/v0.1.0
