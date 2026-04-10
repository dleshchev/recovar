#!/bin/bash
# Build the Docker container and save it as a tarball on the shared filesystem.
# This should be run ONCE before submitting multi-node jobs.
#
# Usage: bash scripts/build_and_save_container.sh [tarball_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARBALL_PATH="${1:-${SCRIPT_DIR}/recovar_container.tar}"

echo "Building Docker container..."
bash "${SCRIPT_DIR}/scripts/build_container.sh"

echo "Saving container to tarball: ${TARBALL_PATH}"
docker save recovar:latest -o "${TARBALL_PATH}"

echo "Tarball saved: $(ls -lh "${TARBALL_PATH}" | awk '{print $5}')"
echo "Done. Multi-node jobs will load from this tarball."
