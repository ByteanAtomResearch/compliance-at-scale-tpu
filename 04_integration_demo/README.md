# Module 4: Integration Demo

This module connects the TPU batch evaluator (Module 2) to rai-checklist-cli's existing report formats. It runs a before/after comparison to make the throughput difference tangible.

## What it does

1. **Sequential baseline (simulated):** Estimates how long it would take to evaluate all 50 records through a hosted API, one call at a time. Each call is simulated at 0.8 seconds (typical for a mid-tier hosted LLM).

2. **TPU batch evaluation:** Runs the same evaluation through vLLM's offline batch interface on TPU (the Module 2 approach).

3. **Report generation:** Converts the batch results into three formats:
   - `evaluation_report.json` (full structured data)
   - `evaluation_report.md` (Markdown checklist, compatible with rai-checklist-cli)
   - `evaluation_report.yaml` (YAML, compatible with rai-checklist-cli's validation)

4. **Comparison table:** Prints a side-by-side throughput comparison.

## Running it

```bash
make demo
```

Or directly:

```bash
python 04_integration_demo/integration_demo.py \
    --input sample_data/llm_outputs.jsonl \
    --model google/gemma-4-E4B-it
```

Add `--skip-sequential` to skip the simulated baseline and just run the TPU evaluation.

## This is a demo

This script is intentionally minimal. It is a teaching tool that shows how batch TPU inference results can feed into an existing compliance reporting pipeline, with the goal being to show what's possible rather than provide a production-ready integration.
