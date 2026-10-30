.DEFAULT_GOAL := check
.PHONY: install fix architecture lint type test check verify

install:  ## Sync the dev environment
	uv sync --locked --extra dev

fix:  ## Auto-format and apply safe lint fixes
	uv run --locked ruff format .
	uv run --locked ruff check --fix .

architecture:  ## Check dependency boundaries and show repair guidance
	uv run --locked lint-imports --no-logo --cache-dir .cache/import-linter

lint:  ## Lint and format-check (no changes)
	uv run --locked ruff check .
	uv run --locked ruff format --check .

type:  ## Strict type-check the package, Modal entrypoints, scripts, and tests
	uv run --locked basedpyright
	uv run --locked basedpyright --project .agents/skills/verify-esme-posttrain

test:  ## Run tests in parallel, including the full CPU stage chain
	uv run --locked pytest -n auto

check:  ## The gate: architecture, lint, types, and tests in parallel
	@$(MAKE) -j4 architecture lint type test

verify:  ## Drive the real stage chain and launch entrypoints offline (ARGS=all|list|control|<drive>)
	.agents/skills/verify-esme-posttrain/verify $(ARGS)
