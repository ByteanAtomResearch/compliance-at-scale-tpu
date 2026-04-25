# Module 1: Environment Setup

This module gets you from zero to a running vLLM instance on Cloud TPU v5e.

## Two installation paths

| Path | Who it's for | Time to ready |
|------|-------------|---------------|
| **Docker (recommended)** | Most users. Reproducible, tested. | ~10 minutes + model download |
| **Source build (advanced)** | Contributors, custom commits. | ~20 minutes + model download |

## Quick start (Docker)

```bash
export PROJECT_ID="your-gcp-project"
export ZONE="us-central2-b"
export HF_TOKEN="hf_..."
bash 01_setup/provision_tpu.sh
```

This creates a Cloud TPU v5e-4 VM and starts the `vllm/vllm-tpu:gemma4` container.

## Verification

Once inside the VM or container:

```bash
python 01_setup/verify_install.py
```

You should see all four checks pass (TPU chips, JAX backend, vllm-tpu, tpu-inference).

## XLA compilation warmup

The first inference request on TPU triggers XLA compilation and static shape bucketization. This takes **20-30 minutes** on v5e-4 and is completely normal. Compiled graphs are cached to `~/.cache/vllm/xla_cache`, so subsequent runs skip this step entirely.

## Colab TPU fallback

If you lack GCE access, you can run a reduced version of this tutorial on Colab's free TPU v2 runtime. Open [`notebooks/tutorial_colab.ipynb`](../notebooks/tutorial_colab.ipynb) in Colab (the badge at the top of the main README is a direct link) and run the cells in order.

The Colab notebook is a condensed end-to-end path: verification check, 10-record batch evaluation, and results table. It uses `google/gemma-3-1b-it` -- a pure text-only Gemma 3 model (~1B parameters, ~2 GB BF16) that runs reliably via the pip-installed `vllm-tpu`. Throughput lands around 3-5 records/second versus 8-12 records/second on a full v5e-4 setup.

**Why not Gemma 4 in Colab?** `Gemma4ForConditionalGeneration` is not registered in the pip package's JAX model registry, and the vLLM fallback path fails on Gemma 4's quantized audio tower weights. Gemma 4 requires the `vllm/vllm-tpu:gemma4` Docker image on GCE hardware (what the main tutorial uses). The batch-inference code pattern is identical between the two paths.

The tutorial code patterns are identical between the two paths. Only the provisioning and model size change.
