#!/usr/bin/env bash
# Fetch the AR mask dataset (~1.3 GB download, ~6 GB extracted) into ./data/.
# The public repo needs no Hugging Face login.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/data/surya-bench-ar-segmentation"
mkdir -p "${DEST}"
hf download nasa-ibm-ai4science/surya-bench-ar-segmentation --repo-type dataset --local-dir "${DEST}"
if [[ ! -d "${DEST}/data" ]]; then
  echo "==> extracting masks"
  mkdir -p "${DEST}/data"
  tar -xzf "${DEST}/data.tar.gz" -C "${DEST}/data"
fi
echo "Masks ready in ${DEST} ($(find "${DEST}/data" -name '*.h5' | wc -l | tr -d ' ') files)"
