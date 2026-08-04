# Mass-Parallelized Compliance: Scaling RAI Checks with vLLM Batch Inference on Cloud TPU

[![Part 1: Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByteanAtomResearch/compliance-at-scale-tpu/blob/v1.0-part1/notebooks/tutorial_colab.ipynb) Part 1: single-output eval
[![Part 2: Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByteanAtomResearch/compliance-at-scale-tpu/blob/main/notebooks/trajectory_colab.ipynb) Part 2: trajectory eval demo
[![CI](https://github.com/ByteanAtomResearch/compliance-at-scale-tpu/actions/workflows/ci.yml/badge.svg)](https://github.com/ByteanAtomResearch/compliance-at-scale-tpu/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

A hands-on, reproducible tutorial for ML practitioners who want to run Responsible AI (RAI) compliance checks at scale using vLLM offline batch inference and an online API server on Cloud TPU v5e.

**No Cloud TPU?** The two Colab badges above run on Colab's free TPU runtime using `google/gemma-3-1b-it`: Part 1 is the condensed single-output evaluation, Part 2 is the trajectory-evaluation demo. The Part 1 badge is pinned to `v1.0-part1`, so the code cannot move underneath the published post. The Part 2 badge tracks `main` and gets pinned to `v2.0-part2` on publication day, once the schema, prompt, and band freeze in `05_trajectory_eval/FROZEN.md` is complete. The patterns are identical to the main tutorial; only the model and hardware differ.

> **Heads up on Colab quotas**: free-tier Colab gates TPU access pretty aggressively. If you see "Cannot connect to TPU backend due to usage limits," you've exhausted your daily allocation. Wait 24 hours for the rolling reset, switch Google accounts, or consider [Kaggle Notebooks](https://www.kaggle.com/code) which offer 30 hours/week of TPU v3-8 free.
>
> **Colab vs GCE model split**: The Colab notebook uses `google/gemma-3-1b-it` (Gemma 3, text-only, works via `pip install vllm-tpu`). The main tutorial uses `google/gemma-4-E4B-it` via the `vllm/vllm-tpu:gemma4` Docker image on GCE. Gemma 4 is not supported on the pip path because `Gemma4ForConditionalGeneration` is not yet in pip's JAX registry and its quantized audio weights break the fallback loader.

This tutorial uses [rai-checklist-cli](https://github.com/ByteanAtomResearch/rai-checklist-cli) as a real-world case study. That package is a CLI tool for generating and validating Responsible AI compliance checklists (Markdown, YAML, JSON) across the AI/ML lifecycle. It covers stages like data privacy, ethical considerations, and deployment monitoring, and its YAML/JSON output can gate CI/CD pipelines.

The tutorial shows how the sequential, per-record evaluation pattern that rai-checklist-cli expects transforms into a mass-parallelized batch pipeline that processes 50 records across 3 heuristics in a single forward pass, then feeds the results directly back into rai-checklist-cli's existing report formats.

## Why this matters

Compliance evaluation has traditionally been a sequential, rate-limited bottleneck. You feed each LLM output through a judge model, wait for a response, log the verdict, and move to the next record. For small datasets that's fine, and at scale it falls apart. Batch inference on TPU sidesteps this entirely by fusing hundreds of judge calls into a single, vectorized forward pass.

## Architecture

```mermaid
flowchart LR
    subgraph Input[Input Data]
        JSONL[llm_outputs.jsonl<br/>50 records]
    end

    subgraph Eval[TPU Batch Evaluation]
        PROMPTS[Prompt Builder<br/>50 x 3 = 150 prompts]
        VLLM[vLLM + tpu-inference<br/>Gemma 4 E4B-it]
        TPU[Cloud TPU v5e-4<br/>4 chips, single host]
        VLLM --> TPU
        TPU --> VLLM
    end

    subgraph Output[Structured Output]
        JSON[batch_results.json]
        MD[evaluation_report.md]
        YAML[evaluation_report.yaml]
        RAI[rai-checklist-cli<br/>validate / report]
    end

    subgraph Serve[Online API Server]
        SERVER[vllm serve<br/>OpenAI-compatible]
        CLIENT[asyncio + httpx<br/>concurrent client]
        CLIENT --> SERVER
        SERVER --> CLIENT
    end

    JSONL --> PROMPTS
    PROMPTS --> VLLM
    VLLM --> JSON
    JSON --> MD
    JSON --> YAML
    MD --> RAI
    YAML --> RAI
    JSONL -.-> CLIENT
    SERVER -.-> JSON

    style TPU fill:#ffd700,stroke:#333,stroke-width:2px
    style VLLM fill:#87ceeb,stroke:#333
    style SERVER fill:#87ceeb,stroke:#333
```

The tutorial walks through two parallel paths to the same destination. The offline batch path (Module 2) is the bulk-processing workhorse. The online server path (Module 3) covers real-time, streaming workloads. Module 4 stitches both into rai-checklist-cli's existing report pipeline.

## What you'll learn

By the end of this tutorial you'll know how to:

- Provision a Cloud TPU v5e-4 VM and run vLLM via Docker
- Write an offline batch evaluation script that processes 150 judge prompts in a single vLLM call
- Launch an OpenAI-compatible API server on TPU and hit it from async Python clients
- Use Gemma 4's native structured JSON output and vLLM structured outputs to eliminate fragile response parsing
- Plug the results into existing compliance reports (Markdown, YAML, JSON)

## Prerequisites

- A Google Cloud project with TPU API enabled
- Quota for Cloud TPU v5e in a supported zone (e.g., `us-central2-b`)
- A Hugging Face account with access to `google/gemma-4-E4B-it`
- Python 3.11+ and [uv](https://docs.astral.sh/uv/) on your local machine
- Basic comfort with the command line and Python

If you lack GCE access, see the Colab TPU v2 fallback note in `01_setup/README.md`. You'll lose throughput compared to v5e, and the code remains the same.

## Repository structure

```
.
├── README.md                            (this file)
├── Makefile                             (one-command runners)
├── pyproject.toml                       (dependency pins for uv)
├── LICENSE                              (Apache 2.0)
├── 01_setup/                            Module 1: Environment
│   ├── provision_tpu.sh                 → GCE TPU VM + Docker image
│   ├── install_from_source.sh           → Advanced: build from tpu-inference
│   ├── verify_install.py                → Smoke test
│   └── README.md
├── 02_offline_batch/                    Module 2: Offline Batch
│   ├── batch_rai_eval.py                → Main script
│   └── README.md
├── 03_online_server/                    Module 3: Online Server
│   ├── start_server.sh                  → Launch vllm serve
│   ├── client_single.py                 → One request demo
│   ├── client_concurrent.py             → asyncio + httpx client
│   └── README.md
├── 04_integration_demo/                 Module 4: rai-checklist-cli bridge
│   ├── integration_demo.py
│   └── README.md
├── 05_trajectory_eval/                  Module 5: Trajectory eval (Part 2)
│   ├── schema.py                        → Record + verdict schemas, hashes
│   ├── canonicalize.py                  → Raw agent log → bounded record
│   ├── corpus.py                        → Corpus loading + guardrails
│   ├── generate_candidates.py           → Tier-2 candidates → staging only
│   ├── adjudicate.py                    → The only path into the corpus
│   ├── rules_baseline.py                → Deterministic checker
│   ├── prompts.py                       → Versioned judge prompts
│   ├── bands.py                         → Token distribution → XLA bands
│   ├── batch_trajectory_eval.py         → Main TPU script
│   ├── score.py                         → Judge vs rules (no TPU)
│   ├── freeze.py + FROZEN.md            → Schema/prompt/band freeze record
│   └── README.md
├── notebooks/
│   ├── tutorial_colab.ipynb             → Part 1 Colab fallback
│   └── trajectory_colab.ipynb           → Part 2 free-tier demo
├── sample_data/
│   ├── llm_outputs.jsonl                → 50 test records
│   ├── expected_output_sample.json      → Reference output
│   ├── staging/                         → Candidate trajectories (pre-adjudication)
│   └── trajectories.jsonl               → Scored corpus (adjudicated only)
└── tests/
    ├── test_pure_functions.py           → Unit tests (no TPU required)
    └── test_trajectory_*.py             → Module 5 unit tests
```

## Quick start

Once you're inside a provisioned TPU VM with the `vllm/vllm-tpu:gemma4` container running:

```bash
# Verify the environment
make verify

# Run the offline batch evaluation (Module 2)
make batch

# Quick smoke test on 5 records (skips the full 50-record run)
make batch ARGS="--limit 5"

# Or launch the online API server (Module 3)
make serve     # in one terminal
make client    # in another terminal

# End-to-end demo with rai-checklist-cli formats (Module 4)
make demo

# Part 2: trajectory evaluation (Module 5)
make trajectory ARGS="--band all --repeat 3"   # timed sweep on TPU
make score                                     # judge vs rules baseline (no TPU)
make bands                                     # token distribution + band proposal (no TPU)

# Run unit tests (no TPU required)
make test
```

## A warning about XLA compilation

The first time you run any vLLM workload on a fresh TPU, XLA compiles the model graph for your specific chip topology and batch shapes. **This takes 20-30 minutes on v5e-4.** During compilation you'll see JAX logs streaming in the container; nothing is broken, it's just working.

Compiled graphs cache to `~/.cache/vllm/xla_cache` on disk. Your second run starts inference in seconds. If you rebuild the container or change the batch shape, you'll trigger a fresh compilation.

Budget for this in your testing timeline. Many first-time TPU users kill the process during compilation thinking it's stuck, then redo the 30-minute wait on every retry.

## Dependency notes

This tutorial uses the `vllm-tpu` package, which is a separate PyPI package from `vllm`. The TPU backend is powered by [tpu-inference](https://github.com/vllm-project/tpu-inference), a unified JAX+PyTorch plugin that replaced the legacy PyTorch/XLA-only code path in vLLM v0.5.x/v0.6.x.

```bash
# Correct for TPU:
uv pip install vllm-tpu

# Wrong (that's the GPU/CUDA package):
pip install vllm
```

The Docker image `vllm/vllm-tpu:gemma4` bundles the right versions and saves you from dependency resolution headaches. If you need to build from source for a specific commit, `01_setup/install_from_source.sh` walks through the tpu-inference pin-and-checkout flow.

## Expected output

After running `make batch` on the sample data, you'll see:

```
Heuristic Results Summary
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Heuristic             ┃ Flagged ┃ Clean ┃ Parse Errors ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━┩
│ Pii Data Leakage      │       8 │    42 │            0 │
│ Jailbreak Override    │       7 │    43 │            0 │
│ Tone Stereotyping     │      10 │    40 │            0 │
└───────────────────────┴─────────┴───────┴──────────────┘
```

Plus a throughput table showing records-per-second. A cold v5e-4 with compilation cache hit typically lands in the 8-12 records/sec range on this 50-record sample. For larger batches the numbers climb as TPU utilization improves.

The full JSON report drops to `results/batch_results.json`, with one entry per record containing all three heuristic verdicts.

## Interpreting your results

The JSON report has three top-level keys: `metadata`, `summary`, and `results`. Here's how to read each one.

**`summary`** is the quickest read. For each of the three heuristics it lists counts of `flagged`, `clean`, and `parse_errors`, plus the list of `flagged_ids`. If a heuristic shows a non-zero `parse_errors` count, the model returned something that failed JSON parsing even with guided decoding active (rare, usually a truncated response from hitting `max_tokens`).

**`results`** is the per-record detail. Each entry has:

- `id` - matches the input record's id
- `source` - the origin label from your input data (e.g., `customer_service_bot`)
- `text_preview` - the first 100 characters of the original text
- `evaluations` - a dict with one entry per heuristic, where each entry contains the structured verdict (e.g., `detected`, `types`, `evidence` for PII)

A typical PII detection entry looks like this:

```json
{
  "detected": true,
  "types": ["phone_number", "email"],
  "evidence": "Contains phone number (555-0142) and email (user@example.com)"
}
```

When `detected` is true, treat the `types` and `evidence` fields as a human-readable audit trail you can surface in a compliance dashboard or route to a reviewer. When `detected` is false, the record passed that heuristic and the other fields can be safely ignored.

The three heuristics are independent: a record can trip all three, exactly one, or none. Records that trip multiple heuristics often warrant the most attention in downstream review.

## Part 2: Trajectory evaluation

Part 1 evaluated single LLM outputs. Part 2 (`05_trajectory_eval/`) evaluates the multi-step execution trajectory that produced an output, against six violation states:

| State | Detects |
|---|---|
| `scope_violation` | Actions outside the assigned goal or boundaries |
| `authorization_bypass` | A required approval gate that never passed |
| `specification_gaming` | Satisfying the metric by defeating its intent |
| `sensitive_state_exposure` | Credentials or protected state crossing a boundary |
| `unsafe_continuation` | Persisting after a stop signal, denial, or error |
| `misreported_state` | Agent self-report diverging from telemetry |

A Gemma judge under vLLM structured outputs is compared per state against a deterministic rules baseline, on a human-adjudicated corpus with length-banded batching for the XLA static-shape model. See [05_trajectory_eval/README.md](05_trajectory_eval/README.md) for the module walkthrough, the adjudication protocol, the truncation design, and the instrumentation dependency table.

Measured results (throughput by band, judge versus rules per state, instrumentation dependency, hosted break-even) are `[MEASURED: pending]` until the TPU runs complete; they publish with the Part 2 article and land in the module README's results table.

## Citing this tutorial

If this tutorial helped your work, a star on the [rai-checklist-cli repo](https://github.com/ByteanAtomResearch/rai-checklist-cli) is appreciated. For academic citations:

<!-- TODO(noble): set year to Part 1's actual publication year and update the
     citation key (ackerson2025...) to match. Deferred at Gate 0: do not guess. -->
```bibtex
@misc{ackerson2025parallelizedcompliance,
  author = {Ackerson, Noble},
  title  = {Mass-Parallelized Compliance: Scaling RAI Checks with vLLM on Cloud TPU},
  year   = {2025},
  howpublished = {\url{https://github.com/ByteanAtomResearch/compliance-at-scale-tpu}},
}
```

## License

Apache 2.0. See [LICENSE](./LICENSE) for the full text. The case-study project referenced in Module 4 ([rai-checklist-cli](https://github.com/ByteanAtomResearch/rai-checklist-cli)) is MIT-licensed and unaffected.

## Feedback

Open an issue on [rai-checklist-cli](https://github.com/ByteanAtomResearch/rai-checklist-cli/issues) with the `tutorial` label. Contributions and corrections are welcome.
