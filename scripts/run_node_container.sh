#!/bin/bash
# Run a pixi task inside the Docker container on a single SLURM node.
# Called by srun in multi-node jobs. Expects env vars:
#   SCRIPT_DIR, TASK_CMD, SLURM_PROCID, SLURM_NTASKS, SLURM_JOB_ID
set -e

echo "Task ${SLURM_PROCID:-0} of ${SLURM_NTASKS:-1} on $(hostname)"

CONTAINER_IMAGE="recovar:latest"
TARBALL_PATH="${SCRIPT_DIR}/recovar_container.tar"

# Load container from tarball (shared filesystem) if not already present
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^recovar:latest$"; then
    if [ -f "${TARBALL_PATH}" ]; then
        echo "Loading container from tarball on $(hostname)..."
        docker load -i "${TARBALL_PATH}"
    else
        echo "No tarball found. Building container on $(hostname)..."
        bash "${SCRIPT_DIR}/scripts/build_container.sh"
    fi
else
    echo "Container image already present on $(hostname)"
fi

docker run --rm --net host --ipc=host \
    --runtime=nvidia \
    -v "${SCRIPT_DIR}":/workspace \
    -w /workspace \
    --user "$(id -u):$(id -g)" \
    -e "SLURM_PROCID=${SLURM_PROCID}" \
    -e "SLURM_NTASKS=${SLURM_NTASKS}" \
    -e "SLURM_JOB_ID=${SLURM_JOB_ID}" \
    -e "SLURM_NODELIST=${SLURM_NODELIST}" \
    "${CONTAINER_IMAGE}" \
    -c "
        set -e
        echo \"Rank \${SLURM_PROCID}: Installing dependencies...\"
        pixi install

        echo \"Rank \${SLURM_PROCID}: Installing RECOVAR...\"
        pixi run install-recovar

        echo \"Rank \${SLURM_PROCID}: Running task: ${TASK_CMD}\"
        ${TASK_CMD}

        echo \"Rank \${SLURM_PROCID}: Task completed!\"
    "
