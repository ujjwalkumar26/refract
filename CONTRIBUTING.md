# Contributing to refract

Thank you for your interest in contributing to refract! This guide will help you get started.

---

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ujjwalkumar26/refract.git
cd refract

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode with all extras
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

---

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run only unit tests
pytest tests/unit/ -x

# Run only integration tests
pytest tests/integration/ -x -m integration
```

### Code Quality

```bash
# Lint
make lint

# Format
make format

# Type check
make typecheck

# Run everything
make check
```

### Before Submitting a PR

1. **Write tests** for any new functionality
2. **Run `make check`** to ensure lint, types, and tests pass
3. **Update docstrings** for any public API changes
4. **Add a CHANGELOG entry** under `## [Unreleased]`

---

## Project Structure

```
src/refract/
├── search.py          # Main entry point: refract.search()
├── types.py           # All data contracts (dataclasses)
├── analysis/          # Query and space analysis
├── metrics/           # Similarity metrics (cosine, euclidean, etc.)
├── routing/           # Weight routing (heuristic, learned, composite)
├── fusion/            # Score fusion engine
├── embedders/         # Optional embedding providers
└── benchmark/         # Evaluation harness
```

---

## Adding a New Metric

1. Create `src/refract/metrics/your_metric.py`
2. Subclass `BaseMetric` and implement `score()` and `batch_score()`
3. Register it in `src/refract/metrics/registry.py`
4. Add tests in `tests/unit/test_metrics.py`
5. Update the heuristic router rules if appropriate

---

## Adding a New Embedder

1. Create `src/refract/embedders/your_embedder.py`
2. Subclass `BaseEmbedder` and implement `embed()`
3. Add the dependency to `pyproject.toml` under `[project.optional-dependencies]`
4. Use lazy imports — the dependency should only be imported when the embedder is used

---

## Code Style

- **Formatter/Linter:** ruff (configured in `pyproject.toml`)
- **Type checker:** mypy with strict mode
- **Docstrings:** Google style
- **Line length:** 100 characters
- **Python version:** 3.9+ (no walrus operator, no `X | Y` union syntax)

---

## Reporting Bugs

Use the [GitHub Issues](https://github.com/ujjwalkumar26/refract/issues) page with the bug report template.

## Requesting Features

Use the [GitHub Issues](https://github.com/ujjwalkumar26/refract/issues) page with the feature request template.

---

## License

By contributing to refract, you agree that your contributions will be licensed under the MIT License.
