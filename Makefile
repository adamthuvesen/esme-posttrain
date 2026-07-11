.DEFAULT_GOAL := check
.PHONY: install fmt lint type test check

install:  ## Sync the dev environment
	uv sync --extra dev

fmt:  ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint and format-check (no changes)
	uv run ruff check .
	uv run ruff format --check .

type:  ## Check source types
	uv run mypy

test:  ## Run unit tests
	uv run python -m pytest

check: lint type test  ## The gate: lint + format-check + types + unit tests
