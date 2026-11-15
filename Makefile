.DEFAULT_GOAL := check
.PHONY: install fix architecture lint type test check verify slow

install:  ## Sync the dev environment
	uv sync --locked --extra dev

fix:  ## Auto-format and apply safe lint fixes
	uv run --locked ruff format . .agents/skills/verify-esme-posttrain/verify
	uv run --locked ruff check --fix . .agents/skills/verify-esme-posttrain/verify

architecture:  ## Check dependency boundaries and show repair guidance
	uv run --locked lint-imports --no-logo --cache-dir .cache/import-linter

lint:  ## Lint and format-check (no changes)
	uv run --locked ruff check . .agents/skills/verify-esme-posttrain/verify
	uv run --locked ruff format --check . .agents/skills/verify-esme-posttrain/verify

type:  ## Strict type-check the package, Modal entrypoints, scripts, and tests
	uv run --locked basedpyright
	uv run --locked basedpyright --project .agents/skills/verify-esme-posttrain

test:  ## Run tests in parallel, including the full CPU stage chain
	uv run --locked pytest -n auto

check:  ## The gate: architecture, lint, types, and tests in parallel
	@$(MAKE) -j4 architecture lint type test

verify:  ## Drive the real stage chain and launch entrypoints offline (ARGS=all|list|control|<drive>)
	.agents/skills/verify-esme-posttrain/verify $(ARGS)

slow:  ## The slow lane: every verify drive, then the control that must observe FAIL
	.agents/skills/verify-esme-posttrain/verify all
	.agents/skills/verify-esme-posttrain/verify control
