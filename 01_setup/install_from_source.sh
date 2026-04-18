#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Module 1 (Advanced): Build vLLM for TPU from Source
#
# This is the secondary install path for users who need to build from source
# (e.g., to test a specific commit or contribute upstream). Most users should
# use the Docker image instead (see provision_tpu.sh).
#
# The flow:
#   1. Clone the tpu-inference repo (Google's unified JAX+PyTorch TPU backend)
#   2. Read the pinned vLLM commit from tpu-inference's vllm_lkg.version file
#   3. Clone vLLM at that exact commit
#   4. Build vLLM with VLLM_TARGET_DEVICE="tpu"
#
# This replaces the old PyTorch/XLA nightly-wheel workflow from vLLM v0.5.x.
# The tpu-inference plugin now handles all TPU backend wiring.
#
# Prerequisites:
#   - Python 3.12
#   - uv (https://docs.astral.sh/uv/)
#   - Running on a TPU VM with chip access
#
# Usage:
#   bash 01_setup/install_from_source.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/vllm-tpu-build}"

echo "============================================"
echo " vLLM TPU: Build from Source"
echo "============================================"
echo ""
echo "  Build directory: ${WORKDIR}"
echo ""

# ── Step 1: Set up Python environment ────────────────────────────────────────
echo "[1/5] Creating Python 3.12 virtual environment with uv..."
uv venv "${WORKDIR}/.venv" --python 3.12
source "${WORKDIR}/.venv/bin/activate"
echo "  Python: $(python --version)"
echo "  venv:   ${WORKDIR}/.venv"

# ── Step 2: Clone tpu-inference ──────────────────────────────────────────────
echo ""
echo "[2/5] Cloning tpu-inference (Google's unified TPU backend)..."
if [ -d "${WORKDIR}/tpu-inference" ]; then
    echo "  Directory exists, pulling latest..."
    cd "${WORKDIR}/tpu-inference" && git pull
else
    git clone https://github.com/google/tpu-inference.git "${WORKDIR}/tpu-inference"
fi

# ── Step 3: Read the pinned vLLM commit ──────────────────────────────────────
echo ""
echo "[3/5] Reading pinned vLLM commit from tpu-inference..."
VLLM_COMMIT=$(cat "${WORKDIR}/tpu-inference/vllm_lkg.version" | tr -d '[:space:]')
echo "  Pinned vLLM commit: ${VLLM_COMMIT}"
echo ""
echo "  Why pin this commit? The tpu-inference plugin is tested against this"
echo "  specific vLLM version. Using a different commit may produce build"
echo "  errors or runtime failures on TPU."

# ── Step 4: Clone and checkout vLLM at the pinned commit ─────────────────────
echo ""
echo "[4/5] Cloning vLLM and checking out pinned commit..."
if [ -d "${WORKDIR}/vllm" ]; then
    cd "${WORKDIR}/vllm"
    git fetch origin
else
    git clone https://github.com/vllm-project/vllm.git "${WORKDIR}/vllm"
    cd "${WORKDIR}/vllm"
fi
git checkout "${VLLM_COMMIT}"
echo "  Checked out vLLM at: $(git log --oneline -1)"

# ── Step 5: Build vLLM with TPU target ───────────────────────────────────────
echo ""
echo "[5/5] Building vLLM with VLLM_TARGET_DEVICE=tpu..."
echo ""
echo "  This compiles vLLM to use the tpu-inference backend instead of CUDA."
echo "  The build takes a few minutes."
echo ""

cd "${WORKDIR}/tpu-inference"
uv pip install -e .

cd "${WORKDIR}/vllm"
VLLM_TARGET_DEVICE="tpu" uv pip install -e .

echo ""
echo "============================================"
echo " Build Complete"
echo "============================================"
echo ""
echo "  Activate the environment:"
echo "    source ${WORKDIR}/.venv/bin/activate"
echo ""
echo "  Verify the installation:"
echo "    python 01_setup/verify_install.py"
echo ""
echo "  If verification passes, you're ready for Module 2."
echo ""
