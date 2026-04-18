#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Module 3: Launch vLLM's OpenAI-compatible API server on TPU
#
# This starts a persistent HTTP server that exposes the same OpenAI chat
# completions API you'd get from OpenAI's servers, but running locally on
# your TPU VM with Gemma 4 E4B.
#
# Once running, you can hit it with the OpenAI Python client, curl, or any
# tool that speaks the OpenAI API format.
#
# Usage:
#   bash 03_online_server/start_server.sh [MODEL]
#
# Arguments:
#   MODEL  - Hugging Face model ID (default: google/gemma-4-E4B-it)
#
# The server binds to 0.0.0.0:8000 by default.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODEL="${1:-google/gemma-4-E4B-it}"
PORT="${PORT:-8000}"
TPU_CHIPS="${TPU_CHIPS:-4}"

echo "============================================"
echo " vLLM OpenAI-Compatible API Server"
echo "============================================"
echo ""
echo "  Model:    ${MODEL}"
echo "  Port:     ${PORT}"
echo "  TPU chips: ${TPU_CHIPS}"
echo ""
echo "  First request will trigger XLA compilation (20-30 min)."
echo "  Check server readiness at http://localhost:${PORT}/health"
echo ""
echo "  Press Ctrl+C to stop the server."
echo ""

# vllm serve is the CLI entry point that starts the OpenAI-compatible server.
# The TPU backend is selected automatically when running on a TPU VM with
# vllm-tpu installed. No extra flags needed for device selection.
vllm serve "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TPU_CHIPS}" \
    --dtype bfloat16 \
    --max-model-len 4096
