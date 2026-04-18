# ──────────────────────────────────────────────────────────────
# Mass-Parallelized Compliance: vLLM on Cloud TPU
# One-command runners for each tutorial module
# ──────────────────────────────────────────────────────────────

MODEL ?= google/gemma-4-E4B-it
DATA  ?= sample_data/llm_outputs.jsonl
PORT  ?= 8000

.PHONY: help setup verify batch serve client client-concurrent demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Module 1: Environment ──────────────────────────────────
setup: ## Provision TPU VM and pull Docker image
	bash 01_setup/provision_tpu.sh

verify: ## Verify TPU + vLLM installation
	python 01_setup/verify_install.py

# ── Module 2: Offline Batch ────────────────────────────────
batch: ## Run offline batch RAI evaluation on sample data
	python 02_offline_batch/batch_rai_eval.py \
		--model $(MODEL) \
		--input $(DATA) \
		--output results/batch_results.json

# ── Module 3: Online Server ────────────────────────────────
serve: ## Launch vLLM OpenAI-compatible API server
	bash 03_online_server/start_server.sh $(MODEL)

client: ## Send a single RAI evaluation request to the running server
	python 03_online_server/client_single.py \
		--port $(PORT)

client-concurrent: ## Send concurrent RAI evaluation requests
	python 03_online_server/client_concurrent.py \
		--port $(PORT) \
		--input $(DATA)

# ── Module 4: Integration Demo ─────────────────────────────
demo: ## Run end-to-end integration demo with rai-checklist-cli
	python 04_integration_demo/integration_demo.py \
		--input $(DATA) \
		--model $(MODEL)

# ── Utilities ──────────────────────────────────────────────
clean: ## Remove generated results
	rm -rf results/
