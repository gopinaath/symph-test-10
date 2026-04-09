.PHONY: test lint type-check format ci

test:
	PYTHONPATH=. pytest tests/ -v --timeout=30 -k "not e2e"

lint:
	ruff check symphony/ tests/

format:
	ruff format symphony/ tests/

type-check:
	mypy symphony/

ci: lint type-check test
