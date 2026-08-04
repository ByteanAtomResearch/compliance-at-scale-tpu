#!/usr/bin/env bash
# XLA cache push/restore for Part 2 TPU sessions.
#
# Compiled graphs live in ~/.cache/vllm/xla_cache inside the TPU VM. Delete
# the VM without pushing and every band recompiles at 20-30 minutes on the
# next session. Push before every teardown, restore after every provision.
#
# Cache validity is tied to the container image, model, and batch shapes:
# bumping any of those voids the cache. Record the image DIGEST, not the
# tag, in the run metadata.
#
# Usage:
#   export PART2_BUCKET=your-bucket-name
#   bash 05_trajectory_eval/cache_helper.sh push
#   bash 05_trajectory_eval/cache_helper.sh restore

set -euo pipefail

CMD="${1:-}"
BUCKET="${PART2_BUCKET:?set PART2_BUCKET to your GCS bucket name (no gs:// prefix)}"
CACHE_PARENT="${HOME}/.cache/vllm"
PREFIX="gs://${BUCKET}/part2"

case "${CMD}" in
  push)
    if [ ! -d "${CACHE_PARENT}/xla_cache" ]; then
      echo "No cache at ${CACHE_PARENT}/xla_cache; nothing to push." >&2
      exit 1
    fi
    STAMP="$(date +%Y%m%d_%H%M%S)"
    tar czf "/tmp/xla_cache_${STAMP}.tgz" -C "${CACHE_PARENT}" xla_cache
    gsutil cp "/tmp/xla_cache_${STAMP}.tgz" "${PREFIX}/xla_cache_${STAMP}.tgz"
    echo "Pushed ${PREFIX}/xla_cache_${STAMP}.tgz"
    ;;
  restore)
    LATEST="$(gsutil ls "${PREFIX}/xla_cache_"'*.tgz' | sort | tail -1)"
    if [ -z "${LATEST}" ]; then
      echo "No cache archives under ${PREFIX}; first session compiles cold." >&2
      exit 1
    fi
    gsutil cp "${LATEST}" /tmp/xla_cache_restore.tgz
    mkdir -p "${CACHE_PARENT}"
    tar xzf /tmp/xla_cache_restore.tgz -C "${CACHE_PARENT}"
    echo "Restored ${LATEST} into ${CACHE_PARENT}/xla_cache"
    ;;
  *)
    echo "Usage: PART2_BUCKET=<bucket> bash $0 {push|restore}" >&2
    exit 2
    ;;
esac
