# ──────────────────────────────────────────────────────────────
# Mass-Parallelized Compliance: vLLM on Cloud TPU
# One-command runners for each tutorial module
# ──────────────────────────────────────────────────────────────

MODEL       ?= google/gemma-4-E4B-it
DATA        ?= sample_data/llm_outputs.jsonl
PORT        ?= 8000
CONCURRENCY ?= 10

.PHONY: help setup verify batch serve client client-concurrent demo trajectory score bands \
	misreport-review-files misreport-rehearsal misreport-corpus misreport-bands misreport-run parity bridge \
	test clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Module 1: Environment ──────────────────────────────────
setup: ## Provision TPU VM and pull Docker image
	bash 01_setup/provision_tpu.sh

verify: ## Verify TPU + vLLM installation
	python 01_setup/verify_install.py

# ── Module 2: Offline Batch ────────────────────────────────
ARGS ?=
batch: ## Run offline batch RAI evaluation on sample data
	python 02_offline_batch/batch_rai_eval.py \
		--model $(MODEL) \
		--input $(DATA) \
		--output results/batch_results.json \
		$(ARGS)

# ── Module 3: Online Server ────────────────────────────────
serve: ## Launch vLLM OpenAI-compatible API server
	bash 03_online_server/start_server.sh $(MODEL)

client: ## Send a single RAI evaluation request to the running server
	python 03_online_server/client_single.py \
		--port $(PORT)

client-concurrent: ## Send concurrent RAI evaluation requests
	python 03_online_server/client_concurrent.py \
		--port $(PORT) \
		--input $(DATA) \
		--concurrency $(CONCURRENCY)

# ── Module 4: Integration Demo ─────────────────────────────
demo: ## Run end-to-end integration demo with rai-checklist-cli
	python 04_integration_demo/integration_demo.py \
		--input $(DATA) \
		--model $(MODEL)

# ── Module 5: Trajectory Eval (Part 2) ─────────────────────
trajectory: ## Run trajectory eval on TPU (ARGS="--band all --repeat 3")
	python 05_trajectory_eval/batch_trajectory_eval.py \
		--model $(MODEL) \
		$(ARGS)

score: ## Score judge vs rules baseline (no TPU)
	python 05_trajectory_eval/score.py \
		--judge results/trajectory_verdicts.json \
		--baseline results/rules_verdicts.json \
		--labels sample_data/trajectories.jsonl

bands: ## Emit corpus token distribution + band proposal (no TPU)
	python 05_trajectory_eval/bands.py --corpus sample_data/trajectories.jsonl

# ── Module 6: Misreported State on TorchTPU (Part 3) ───────
MISREPORT_MODEL ?= RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic
MISREPORT_DIR   ?= 06_misreported_state_torchtpu
BACKEND         ?=

misreport-review-files: ## Rebuild 06's staged review files from the tier-2 batch (no TPU)
	python $(MISREPORT_DIR)/build_review_files.py

misreport-rehearsal: ## Rehearse every laptop-only 06 step on fixtures; fails on any mismatch (no TPU)
	python $(MISREPORT_DIR)/rehearsal.py

misreport-corpus: ## Concatenate 05's adjudicated corpus and 06's tamper corpus (after Lab C)
	mkdir -p results
	cat sample_data/trajectories.jsonl $(MISREPORT_DIR)/data/tamper_corpus.jsonl > results/06_eval_corpus.jsonl
	wc -l results/06_eval_corpus.jsonl

misreport-bands: ## 26B band report for the 06 eval corpus (no TPU; needs transformers + HF auth)
	python 05_trajectory_eval/bands.py \
		--corpus results/06_eval_corpus.jsonl \
		--tokenizer $(MISREPORT_MODEL) \
		--output results/06_band_report_26b.json

misreport-run: ## Eval-set run on TPU (BACKEND=jax|torchtpu, ARGS="--dtype auto --limit 2 --dry-run")
	$(if $(BACKEND),,$(error BACKEND=jax|torchtpu is required))
	python $(MISREPORT_DIR)/run.py \
		--backend $(BACKEND) \
		--model $(MISREPORT_MODEL) \
		--input results/06_eval_corpus.jsonl \
		--bands-report results/06_band_report_26b.json \
		--eval-set \
		--output results/06_verdicts_$(BACKEND).json \
		$(ARGS)

parity: ## Compare the jax and torchtpu verdicts; refuses on any config difference (no TPU)
	python $(MISREPORT_DIR)/parity.py \
		--left results/06_verdicts_jax.json --left-name jax \
		--right results/06_verdicts_torchtpu.json --right-name torchtpu \
		--labels results/06_eval_corpus.jsonl \
		--output results/06_parity.json

bridge: ## Turn the torchtpu verdicts into rai-checklist-cli Markdown + YAML (no TPU)
	python $(MISREPORT_DIR)/bridge.py \
		--verdicts results/06_verdicts_torchtpu.json \
		--markdown results/06_report.md \
		--yaml results/06_report.yaml

# ── Utilities ──────────────────────────────────────────────
test: ## Run unit tests (no TPU hardware required)
	python -m unittest discover -s tests -p "test_*.py" -v

clean: ## Remove generated results
	rm -rf results/
