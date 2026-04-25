#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Module 1: Provision a Cloud TPU v5e-4 VM and launch the vLLM Docker image
#
# This script automates the two-step setup:
#   1. Create a GCE TPU VM (v5e with 4 chips, single host)
#   2. SSH in and start the vllm/vllm-tpu:gemma4 Docker container
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - A GCP project with TPU API enabled
#   - Sufficient quota for TPU v5e in your chosen zone
#
# Usage:
#   export PROJECT_ID="your-gcp-project"
#   export ZONE="us-central2-b"          # or any zone with v5e availability
#   export HF_TOKEN="hf_..."            # Hugging Face token for gated models
#   bash 01_setup/provision_tpu.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your GCP project}"
ZONE="${ZONE:-us-central2-b}"
TPU_NAME="${TPU_NAME:-vllm-tpu-tutorial}"
TPU_TYPE="v5litepod-4"
RUNTIME_VERSION="v2-alpha-tpuv5-lite"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN for Hugging Face model access}"

DOCKER_IMAGE="vllm/vllm-tpu:gemma4"
MODEL="google/gemma-4-E4B-it"

echo "============================================"
echo " Cloud TPU v5e-4 Provisioning"
echo "============================================"
echo ""
echo "  Project:  ${PROJECT_ID}"
echo "  Zone:     ${ZONE}"
echo "  VM Name:  ${TPU_NAME}"
echo "  TPU Type: ${TPU_TYPE}"
echo ""

# ── Step 1: Create the TPU VM ────────────────────────────────────────────────
echo "[1/3] Creating TPU VM..."
echo ""
echo "  This provisions a single-host v5e with 4 TPU chips."
echo "  If the VM already exists, this step will be skipped."
echo ""

if gcloud compute tpus tpu-vm describe "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" &>/dev/null; then
    echo "  TPU VM '${TPU_NAME}' already exists. Skipping creation."
else
    # Remove --preemptible to create an on-demand VM (recommended for tutorials
    # where XLA compilation takes 20-30 min on first run -- preemptible VMs can
    # be killed mid-compilation). Add --preemptible back here if you want to
    # reduce cost and are willing to handle restarts.
    gcloud compute tpus tpu-vm create "${TPU_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --accelerator-type="${TPU_TYPE}" \
        --version="${RUNTIME_VERSION}"

    echo "  TPU VM created successfully."
fi

# ── Step 2: Pull and run the Docker container ────────────────────────────────
echo ""
echo "[2/3] Launching vLLM Docker container on TPU VM..."
echo ""
echo "  Image:  ${DOCKER_IMAGE}"
echo "  Model:  ${MODEL}"
echo ""
echo "  The container runs with:"
echo "    --privileged       (required for TPU device access)"
echo "    --net=host         (expose the API server on the host network)"
echo "    --shm-size=16g     (shared memory for model weights)"
echo ""

# Write the HF token to a temp file on the remote VM and pass it via --env-file
# to avoid the token appearing in `docker inspect` output.
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --command="
        # Pull the latest image
        sudo docker pull ${DOCKER_IMAGE}

        # Write the HF token to a restricted env file so it doesn't appear
        # in 'docker inspect' output (safer than passing -e HF_TOKEN=... inline).
        printf 'HF_TOKEN=%s\n' '${HF_TOKEN}' > /tmp/vllm-env
        chmod 600 /tmp/vllm-env

        # Run the container in detached mode
        sudo docker run -d \
            --privileged \
            --net=host \
            --shm-size=16g \
            --name vllm-server \
            --env-file /tmp/vllm-env \
            ${DOCKER_IMAGE} \
            --model ${MODEL}

        rm -f /tmp/vllm-env

        echo ''
        echo 'Container launched. Checking status...'
        sleep 5
        sudo docker ps --filter name=vllm-server --format 'table {{.ID}}\t{{.Status}}\t{{.Ports}}'
    "

# ── Step 3: Print next steps ─────────────────────────────────────────────────
echo ""
echo "[3/3] Done! Your TPU VM is running."
echo ""
echo "============================================"
echo " Next Steps"
echo "============================================"
echo ""
echo "  1. SSH into your VM:"
echo "     gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}"
echo ""
echo "  2. Check container logs (watch for XLA compilation):"
echo "     sudo docker logs -f vllm-server"
echo ""
echo "  3. Run the verification script:"
echo "     python 01_setup/verify_install.py"
echo ""
echo "  IMPORTANT: The first request triggers XLA compilation,"
echo "  which takes 20-30 minutes on v5e-4. Subsequent requests"
echo "  use the disk cache at ~/.cache/vllm/xla_cache and are fast."
echo ""
