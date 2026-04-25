# Module 2: Offline Batch RAI Evaluation

This module is the core of the tutorial. It runs all 50 sample LLM outputs through three RAI heuristics in a single vLLM batch call on TPU, using Gemma 4 E4B as an "LLM-as-a-Judge."

## What it does

For each record in `sample_data/llm_outputs.jsonl`, the script builds three prompts (one per heuristic), sends all 150 prompts to vLLM's offline `LLM` interface in a single batch, and aggregates the results into a structured JSON report.

The three heuristics:

1. **PII / Data Leakage** - Flags unmasked phone numbers, emails, SSNs, credit cards, medical IDs
2. **Jailbreak / Override** - Detects prompt manipulation, DAN attacks, safety bypasses
3. **Tone & Stereotyping** - Catches demographic stereotyping, exclusionary language, bias

## Running it

```bash
make batch
```

Or directly:

```bash
python 02_offline_batch/batch_rai_eval.py \
    --model google/gemma-4-E4B-it \
    --input sample_data/llm_outputs.jsonl \
    --output results/batch_results.json
```

For a quick smoke test that skips the full 50-record run:

```bash
python 02_offline_batch/batch_rai_eval.py \
    --model google/gemma-4-E4B-it \
    --input sample_data/llm_outputs.jsonl \
    --output results/batch_results.json \
    --limit 5
```

## Expected output

The script prints a summary table and throughput metrics to the console, and writes the full evaluation report to `results/batch_results.json`.

## First-run compilation

On the first invocation, XLA compiles the model graph for your specific TPU topology and batch shapes. This takes **20-30 minutes** on v5e-4. The compiled graphs are cached to `~/.cache/vllm/xla_cache`, so the second run starts inference almost immediately.
