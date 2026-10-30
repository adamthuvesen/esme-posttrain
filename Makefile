.DEFAULT_GOAL := check
.PHONY: install fmt architecture lint type test check

install:  ## Sync the dev environment
	uv sync --locked --extra dev

fmt:  ## Auto-format and apply safe lint fixes
	uv run --locked ruff format .
	uv run --locked ruff check --fix .

architecture:  ## Check dependency boundaries and show repair guidance
	uv run --locked lint-imports --no-logo --cache-dir .cache/import-linter

lint:  ## Lint and format-check (no changes)
	uv run --locked ruff check .
	uv run --locked ruff format --check .

type:  ## Check package and Modal entrypoint types
	uv run --locked mypy

test:  ## Run tests, including the full CPU stage chain
	uv run --locked pytest

check: architecture lint type test  ## The gate: architecture + lint + types + tests
