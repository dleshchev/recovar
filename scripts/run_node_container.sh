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

# Rank 0 installs pixi env first (shared filesystem — can't install concurrently).
# Other ranks wait for a marker file before running.
INSTALL_MARKER="${SCRIPT_DIR}/.pixi_install_done_${SLURM_JOB_ID}"

if [ "${SLURM_PROCID}" = "0" ]; then
    echo "Rank 0: Installing pixi env (other ranks will wait)..."
    docker run --rm --net host --ipc=host \
        --runtime=nvidia \
        -v "${SCRIPT_DIR}":/workspace \
        -w /workspace \
        --user "$(id -u):$(id -g)" \
        "${CONTAINER_IMAGE}" \
        -c "
            set -e
            echo 'Installing dependencies...'
            pixi install
            echo 'Installing RECOVAR...'
            pixi run install-recovar
            echo 'Install complete.'
        "
    touch "${INSTALL_MARKER}"
    echo "Rank 0: Install marker written."
else
    echo "Rank ${SLURM_PROCID}: Waiting for rank 0 to finish install..."
    for i in $(seq 1 600); do
        [ -f "${INSTALL_MARKER}" ] && break
        sleep 1
    done
    if [ ! -f "${INSTALL_MARKER}" ]; then
        echo "ERROR: Timed out waiting for rank 0 install (600s)"
        exit 1
    fi
    echo "Rank ${SLURM_PROCID}: Install marker found, proceeding."
fi

# Now run the actual task (pixi env already installed on shared filesystem)
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
        echo \"Rank \${SLURM_PROCID}: Running task: ${TASK_CMD}\"
        ${TASK_CMD}
        echo \"Rank \${SLURM_PROCID}: Task completed!\"
    "

# Cleanup marker (rank 0 only, after all tasks done)
if [ "${SLURM_PROCID}" = "0" ]; then
    rm -f "${INSTALL_MARKER}"
fi
