# Single BLAS/PyTorch thread per worker.
#
# The parallel runner uses process-level parallelism: many single-threaded
# workers beat one multi-threaded worker, because these experiments are many
# small independent runs rather than one large matrix workload.
#
# Export before Python starts, otherwise the BLAS runtime has already sized
# its thread pool and the setting has no effect.
export OMP_NUM_THREADS ?= 1
export VECLIB_MAXIMUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export NUMEXPR_NUM_THREADS ?= 1

UV ?= uv
PY := $(UV) run

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: sync
sync: ## Resolve and install the environment
	$(UV) sync --all-extras

.PHONY: lock
lock: ## Refresh uv.lock
	$(UV) lock

.PHONY: fmt
fmt: ## Format code
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests

.PHONY: lint
lint: ## Lint without modifying
	$(PY) ruff format --check src tests
	$(PY) ruff check src tests

.PHONY: types
types: ## Type-check
	$(PY) mypy src

.PHONY: test
test: ## Run the fast test suite (excludes `slow` marker)
	$(PY) pytest

.PHONY: test-slow
test-slow: ## Run the whole suite including training tests
	$(PY) pytest -m "slow or not slow"

.PHONY: check
check: lint types test ## Everything that must pass before commit

.PHONY: check-all
check-all: lint types test-slow ## Everything, including training tests

.PHONY: videos
videos: ## Render the latency-ladder rollout videos into results/videos/
	$(PY) python scripts/render_videos.py

.PHONY: site
site: ## Build the local visualisation site (videos + metrics) into site/
	$(PY) python scripts/build_site.py

.PHONY: site-serve
site-serve: ## Serve the site at http://localhost:8000
	$(UV) run python -m http.server 8000 --directory site

.PHONY: verify-results
verify-results: ## Provenance-check committed results (override with SWEEP=)
	$(PY) latency-ml verify --root results --sweep-id "$(or $(SWEEP),)" --verbose

.PHONY: bootstrap
bootstrap: ## Record the resolved dependency stack (PLAN.md section 19)
	@$(PY) python -c "import json, sys, importlib.metadata as md; \
		names = ['gymnasium','numpy','torch','box2d','pygame-ce']; \
		out = {'python': sys.version.split()[0]}; \
		[out.__setitem__(n, md.version(n)) for n in names]; \
		print(json.dumps(out, indent=2))"

.PHONY: clean
clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +