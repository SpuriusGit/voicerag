.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Light backends: no model downloads, no GPU. Used by tests and `make demo`.
LIGHT_ENV := VOICERAG_RAG__EMBEDDER=hashing \
             VOICERAG_RAG__RERANKER=noop \
             VOICERAG_LLM__BACKEND=echo \
             VOICERAG_LLM__MODEL=echo \
             VOICERAG_STT__BACKEND=stub \
             VOICERAG_RAG__STORE_PATH=storage/index-light

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: install
install: $(VENV)/bin/activate ## Install runtime + dev dependencies (CPU only)
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PIP) install -e .

.PHONY: install-ml
install-ml: install ## Also install the GPU/ML stack (torch, transformers, whisper, peft)
	$(PIP) install -r requirements-ml.txt

.PHONY: test
test: ## Run the test suite
	$(PY) -m pytest -q

.PHONY: cov
cov: ## Run tests with a coverage report
	$(PY) -m pytest --cov=voicerag --cov-report=term-missing --cov-report=html

.PHONY: lint
lint: ## Lint and type-check
	$(VENV)/bin/ruff check src tests finetune scripts
	$(VENV)/bin/ruff format --check src tests finetune scripts
	$(VENV)/bin/mypy || true

.PHONY: format
format: ## Auto-format and auto-fix
	$(VENV)/bin/ruff check --fix src tests finetune scripts
	$(VENV)/bin/ruff format src tests finetune scripts

.PHONY: ingest
ingest: ## Build the index with the configured embedder
	$(VENV)/bin/voicerag ingest --corpus data/corpus

.PHONY: serve
serve: ## Run the API on :8000
	$(VENV)/bin/voicerag serve --reload

.PHONY: eval
eval: ## Run the evaluation harness (add JUDGE=1 for LLM-as-a-judge)
	$(VENV)/bin/voicerag eval $(if $(JUDGE),--judge,)

.PHONY: bench
bench: ## Measure per-stage latency and peak GPU/RAM
	$(VENV)/bin/voicerag bench --runs 20 --out runs/bench/latest.json

.PHONY: ablation
ablation: ## Compare retrieval configurations and write docs/EXPERIMENTS table
	$(PY) scripts/run_ablation.py

.PHONY: demo
demo: ## End-to-end demo with zero downloads (hashing + echo + stub)
	$(LIGHT_ENV) $(VENV)/bin/voicerag ingest --corpus data/corpus
	$(LIGHT_ENV) $(VENV)/bin/voicerag search "how much vram does the rtx 4060 have"
	$(LIGHT_ENV) $(VENV)/bin/voicerag eval --out runs/demo --run-id demo

.PHONY: prompts
prompts: ## List the prompt registry
	$(VENV)/bin/voicerag prompts list

.PHONY: docker-build
docker-build: ## Build the CPU image
	docker build -f docker/Dockerfile -t voicerag:cpu .

.PHONY: docker-up
docker-up: ## Start the full stack (api + ollama + prometheus + grafana)
	docker compose -f docker/docker-compose.yml up -d --build

.PHONY: docker-down
docker-down: ## Stop the stack
	docker compose -f docker/docker-compose.yml down

.PHONY: finetune-data
finetune-data: ## Build the LoRA training set from the corpus
	$(PY) finetune/prepare_dataset.py

.PHONY: finetune
finetune: ## Train a LoRA adapter (requires make install-ml and a GPU)
	$(PY) finetune/train_lora.py --config finetune/configs/lora_qwen3b.yaml

.PHONY: clean
clean: ## Remove caches, indexes and run artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage storage runs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
