.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---- setup -----------------------------------------------------------------

$(VENV):
	python3 -m venv $(VENV)

.PHONY: install
install: $(VENV) ## Install backend deps (editable) and frontend deps
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev,llm]"
	cd web && npm install --no-audit --no-fund

# ---- run -------------------------------------------------------------------

.PHONY: api
api: ## Run the API gateway (includes the inline worker with the memory broker)
	$(VENV)/bin/uvicorn api_gateway.app:app --reload --port 8000

.PHONY: web
web: ## Run the canvas
	cd web && npm run dev

.PHONY: worker
worker: ## Run the execution worker as a separate process (Redis deployments)
	$(PY) -m orchestrator.worker

# ---- quality ---------------------------------------------------------------

.PHONY: test
test: ## Run the backend test suite (SQLite)
	$(PY) -m pytest -q

.PHONY: test-postgres
test-postgres: ## Run the same suite against PostgreSQL (each test in its own schema)
	@test -n "$(CWAP_TEST_DATABASE_URL)" || { \
		echo "Set CWAP_TEST_DATABASE_URL, e.g."; \
		echo "  make test-postgres CWAP_TEST_DATABASE_URL=postgresql+psycopg://cwap@localhost/cwap"; \
		exit 1; }
	CWAP_TEST_DATABASE_URL="$(CWAP_TEST_DATABASE_URL)" $(PY) -m pytest -q

.PHONY: typecheck
typecheck: ## Typecheck the frontend
	cd web && npx tsc --noEmit

.PHONY: build
build: ## Production build of the frontend
	cd web && npm run build

.PHONY: lint
lint: ## Lint the backend
	$(PY) -m ruff check packages services tests

.PHONY: check
check: test typecheck ## Everything CI runs

# ---- contracts -------------------------------------------------------------

.PHONY: contracts
contracts: ## Re-approve the contract fingerprint lock (do this in a reviewed commit)
	$(PY) -m cwap_contracts.registry

.PHONY: clean
clean: ## Remove local state and build artefacts
	rm -f cwap.sqlite3 cwap.sqlite3-wal cwap.sqlite3-shm dev.sqlite3*
	rm -rf web/.next .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
