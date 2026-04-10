.PHONY: install test test-cov lint format typecheck check clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -x -q

test-cov:
	pytest tests/ --cov=refract --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/refract/

check: lint typecheck test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
