# refract -- Coding Plan (Implemented)

This document describes the architecture and implementation of refract as built.
It serves as the authoritative reference for the codebase.

---

## Repository structure

```
refract/
├── .github/
│   ├── workflows/ci.yml              # GitHub Actions: lint + test matrix (3.9-3.13)
│   └── ISSUE_TEMPLATE/               # Bug report + feature request templates
│
├── src/refract/
│   ├── __init__.py                    # Public API surface (21 exports)
│   ├── _version.py                    # Version: 0.1.0
│   ├── py.typed                       # PEP 561 type marker
│   ├── search.py                      # refract.search() + search_batch()
│   ├── types.py                       # SearchResult, Provenance, QueryProfile, SpaceProfile
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── query_analyzer.py          # Query type detection + feature extraction
│   │   └── space_analyzer.py          # Embedding space geometry profiling
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseMetric ABC
│   │   ├── cosine.py                  # Vectorized cosine similarity
│   │   ├── euclidean.py               # 1/(1+dist) similarity
│   │   ├── mahalanobis.py             # Corpus-fitted, einsum batch scoring
│   │   ├── bm25.py                    # Sparse lexical via rank_bm25
│   │   └── registry.py               # MetricRegistry + DEFAULT_REGISTRY
│   │
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseRouter ABC
│   │   ├── heuristic.py               # HeuristicRouter (12 rule table + adjustments)
│   │   ├── learned.py                 # LearnedRouter (MLP gating, requires torch)
│   │   └── composite.py              # CompositeRouter (weighted blend)
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── weighted.py                # Weighted score fusion + provenance builder
│   │
│   ├── embedders/
│   │   ├── __init__.py                # BaseEmbedder only (lazy imports)
│   │   ├── base.py                    # BaseEmbedder ABC
│   │   ├── sentence_transformers.py   # SentenceTransformerEmbedder
│   │   ├── openai.py                  # OpenAIEmbedder (batched, env key)
│   │   └── cohere.py                  # CohereEmbedder (input_type aware)
│   │
│   └── benchmark/
│       ├── __init__.py
│       ├── harness.py                 # BenchmarkHarness + BenchmarkResult
│       ├── datasets.py                # BeirDataset + CustomDataset
│       └── eval_metrics.py            # recall_at_k, ndcg_at_k, mrr
│
├── tests/
│   ├── conftest.py                    # Shared fixtures (corpus, vectors, queries)
│   ├── unit/
│   │   ├── test_types.py              # Type instantiation + sorting
│   │   ├── test_metrics.py            # All 4 metrics + registry (24 tests)
│   │   ├── test_query_analyzer.py     # Type detection + entropy (18 tests)
│   │   ├── test_space_analyzer.py     # Density, anisotropy (10 tests)
│   │   ├── test_heuristic_router.py   # Weights, adjustments (8 tests)
│   │   └── test_fusion.py            # Fusion, provenance (4 tests)
│   └── integration/
│       └── test_search_api.py        # Full pipeline tests (22 tests)
│
├── examples/
│   ├── quickstart.py                  # 5-line usage demo
│   ├── rag_pipeline.py                # RAG retrieval step
│   ├── code_search.py                 # Code similarity + query type detection
│   ├── custom_metric.py               # Custom BaseMetric implementation
│   ├── compare_cosine.py              # Side-by-side vs vanilla cosine
│   ├── benchmark_demo.py              # BenchmarkHarness evaluation
│   └── vector_db_integration.py       # FAISS/Qdrant integration pattern
│
├── samples/
│   └── mini_corpus.json               # 20 docs + 5 queries with relevance labels
│
├── pyproject.toml                     # hatchling, src layout, extras, tool configs
├── Makefile                           # Dev commands: test, lint, format, typecheck
├── .pre-commit-config.yaml            # ruff + pre-commit hooks
├── .gitignore
├── LICENSE                            # MIT
├── CONTRIBUTING.md                    # Dev setup + workflow guide
├── CHANGELOG.md                       # Keep a Changelog format
├── README.md                          # Full docs with badges, diagrams, examples
└── ARCHITECTURE.md                    # Design rationale + pipeline flow
```

---

## Public API surface (`__init__.py`)

```python
# Main functions
refract.search(query, corpus, *, top_k, embedder, router, metrics) -> list[SearchResult]
refract.search_batch(queries, corpus, *, top_k, embedder, router, metrics) -> list[list[SearchResult]]

# Types
SearchResult, Provenance, MetricScore, QueryProfile, SpaceProfile

# Metrics
BaseMetric, CosineMetric, EuclideanMetric, MahalanobisMetric, BM25Metric
MetricRegistry, DEFAULT_REGISTRY

# Routing
BaseRouter, HeuristicRouter, CompositeRouter

# Embedders (base only -- implementations via direct import)
BaseEmbedder

# Benchmark
BenchmarkHarness, BenchmarkResult
```

---

## search() internal flow

```
1. Resolve inputs (str/ndarray for query and corpus)
2. Embed text if needed (embedder or TF-IDF fallback)
3. Compute quick cosine scores (cheap, used for analysis)
4. query_profile = analyze_query(text, vector, candidates, cosine_scores)
5. space_profile = analyze_space(candidates, cosine_scores)
6. Resolve and fit metrics (Mahalanobis.fit(), BM25.fit_text())
7. weights = router.route(query_profile, space_profile, metric_names)
8. results = fuse(profiles, candidates, weights, metrics)
9. Return top_k SearchResult list
```

---

## Quality standards (verified)

- [x] Type annotations on all public functions and classes
- [x] Google-style docstrings on all public API surfaces
- [x] `ruff check` passes with zero errors
- [x] All 86 tests pass
- [x] All 7 examples run end-to-end
- [x] Optional imports wrapped in try/except with install hints
- [x] `py.typed` marker for PEP 561
- [x] `__all__` defined on all packages
- [x] No global mutable state except `DEFAULT_REGISTRY` (documented)
- [x] Frozen dataclasses with `__slots__` for types

---

## Dependency tiers

| Tier | Dependencies | Install command |
|---|---|---|
| Core | numpy, scipy, scikit-learn, rank_bm25 | `pip install refract-search` |
| sentence-transformers | + sentence-transformers | `pip install "refract-search[sentence-transformers]"` |
| openai | + openai | `pip install "refract-search[openai]"` |
| cohere | + cohere | `pip install "refract-search[cohere]"` |
| learned | + torch | `pip install "refract-search[learned]"` |
| benchmark | + datasets | `pip install "refract-search[benchmark]"` |
| dev | + pytest, pytest-cov, ruff, mypy, pre-commit | `pip install "refract-search[dev]"` |
| all | everything | `pip install "refract-search[all]"` |

---

## Version targets

| Version | Contents | Status |
|---|---|---|
| 0.1.0 | Core API, heuristic router, built-in metrics, TF-IDF fallback, benchmark harness, embedder interfaces, examples, tests | **Done** |
| 0.2.0 | Learned router training harness, BEIR benchmark results | Planned |
| 0.3.0 | Performance optimizations, caching layer | Planned |
| 1.0.0 | Stable API, published benchmarks, documentation site | Planned |
