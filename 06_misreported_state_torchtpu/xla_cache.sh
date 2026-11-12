#!/usr/bin/env bash
# XLA cache push/restore for Part 3 TPU sessions, one GCS prefix per backend.
#
# Compiled graphs live in the XLA cache directory on the TPU VM. Delete the VM
# without pushing and every band recompiles at 20-30 minutes on the next
# session. Push before every teardown, restore after every provision.
#
# Why a copy of 05's cache_helper.sh: that script hardcodes the part2 prefix,
# takes no argument for another, and 05 is frozen. Part 3 runs two backends
# whose compiled graphs are not interchangeable, so each gets its own prefix.
# The cache directory follows VLLM_XLA_CACHE_PATH when set, the same variable
# 06's run.py records in the run metadata, and 05's default path otherwise.
# Use the same VLLM_XLA_CACHE_PATH on push and on restore.
#
# Cache validity is tied to the image or venv, the model and the batch shapes:
# bumping any of those voids the cache. Record the image digest (or the venv
# freeze hash) in the run metadata, never a tag.
#
# Usage:
#   export PART3_BUCKET=your-bucket-name
#   bash 06_misreported_state_torchtpu/xla_cache.sh push torchtpu
#   bash 06_misreported_state_torchtpu/xla_cache.sh restore jax

set -euo pipefail

CMD="${1:-}"
BACKEND="${2:-}"
USAGE="Usage: PART3_BUCKET=<bucket> bash $0 {push|restore} {jax|torchtpu}"

case "${BACKEND}" in
  jax|torchtpu) ;;
  *)
    echo "${USAGE}" >&2
    exit 2
    ;;
esac

BUCKET="${PART3_BUCKET:?set PART3_BUCKET to your GCS bucket name (no gs:// prefix)}"
CACHE_DIR="${VLLM_XLA_CACHE_PATH:-${HOME}/.cache/vllm/xla_cache}"
CACHE_PARENT="$(dirname "${CACHE_DIR}")"
CACHE_NAME="$(basename "${CACHE_DIR}")"
PREFIX="gs://${BUCKET}/part3/${BACKEND}"

case "${CMD}" in
  push)
    if [ ! -d "${CACHE_DIR}" ] || [ -z "$(ls -A "${CACHE_DIR}")" ]; then
      echo "No cache at ${CACHE_DIR}; nothing to push." >&2
      exit 1
    fi
    STAMP="$(date +%Y%m%d_%H%M%S)"
    ARCHIVE="/tmp/xla_cache_${BACKEND}_${STAMP}.tgz"
    tar czf "${ARCHIVE}" -C "${CACHE_PARENT}" "${CACHE_NAME}"
    gsutil cp "${ARCHIVE}" "${PREFIX}/xla_cache_${STAMP}.tgz"
    echo "Pushed ${PREFIX}/xla_cache_${STAMP}.tgz (from ${CACHE_DIR})"
    ;;
  restore)
    LATEST="$(gsutil ls "${PREFIX}/xla_cache_"'*.tgz' 2>/dev/null | sort | tail -1 || true)"
    if [ -z "${LATEST}" ]; then
      echo "No cache archives under ${PREFIX}; first session compiles cold." >&2
      exit 1
    fi
    ARCHIVE="/tmp/xla_cache_restore_${BACKEND}.tgz"
    gsutil cp "${LATEST}" "${ARCHIVE}"
    mkdir -p "${CACHE_PARENT}"
    tar xzf "${ARCHIVE}" -C "${CACHE_PARENT}"
    echo "Restored ${LATEST} into ${CACHE_DIR}"
    ;;
  *)
    echo "${USAGE}" >&2
    exit 2
    ;;
esac
