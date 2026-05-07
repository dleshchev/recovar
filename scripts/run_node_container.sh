#!/bin/bash
# Run a pixi task inside the Docker container on a single SLURM node.
# Called by srun in multi-node jobs. Expects env vars:
#   SCRIPT_DIR, TASK_CMD, SLURM_PROCID, SLURM_NTASKS, SLURM_JOB_ID
#
# =============================================================================
# Phase −1 cluster probe findings (2026-04-27, computelab-sc-01, slurm 25.05.5)
# =============================================================================
# Recorded for the OpenMPI + mpi4py migration (RECOVAR_MPI=1 path).
# Source: sbatch job 1975903 — scripts/output/slurm-1975903.{out,err}
#
# 1. SLURM MPI plugins (`srun --mpi=list`):
#       none, pmi2, pmix, cray_shasta   |   pmix_v4 explicitly available.
#    → PMIx is selectable. Default `MpiDefault = (null)`, so srun callers
#      MUST pass `--mpi=pmix` explicitly. PMI2 also works as a fallback.
#
# 2. pyxis / enroot:
#       /bin/enroot present (enroot 3.5.0).
#       /etc/slurm/plugstack.conf includes plugstack.conf.d/pyxis.conf.
#       `srun --help` lists `[pyxis]` flags (--container-image, --container-mounts, ...).
#    → pyxis SPANK plugin is loaded cluster-wide, even though `which pyxis`
#      returns nothing (it is NOT a binary on PATH — it is a slurm plugin).
#    → Phase 0 Layer C uses the **pyxis-available chain**: a single outer
#      `srun --mpi=pmix --container-image=...` per multi-node job. No nested
#      srun, no mpirun-ssh fallback, no sshd in the container.
#
# 3. Fabric:
#       Login/dev node x11-0436 has /sys/class/infiniband/{irdma0,irdma1}
#       (Intel iRDMA, RoCEv2 over Ethernet — not classic IB).
#       Compute nodes ipp1-2139/2143 (a100-80gb-pcie partition) have no
#       /sys/class/infiniband entries. ibstat is absent on both.
#    → No assumption of IB; let UCX/OB1 BTL pick the best available transport
#      (typically TCP/RoCE depending on partition). No --mca tunings hardcoded
#      until Phase 0 Layer D shows they are needed.
#
# 4. 2-node PMIx hello world (sbatch job 1975903 across ipp1-{2139,2143}):
#       `srun --mpi=pmix --ntasks-per-node=1 hostname`     → both ranks reported, exit 0
#       `srun --mpi=pmi2 --ntasks-per-node=1 hostname`     → both ranks reported, exit 0
#    → PMIx (and PMI2) launch correctly across nodes on bare metal.
#
# 5. Open question deferred to Phase 0 Layer A: `MPI.COMPLEX` symbol availability
#    in mpi4py against apt OpenMPI (Ubuntu 22.04 → openmpi 4.1.x) — answered by
#    Layer A's mpi4py hello-world script. If absent, Phase 3 falls back to
#    viewing complex64 as float32×2.
#
# =============================================================================
# Phase 0 Layer A intermediate findings (jobs 1976756, 1976974, 1977002)
# =============================================================================
# Cluster-specific reality check on the plan's two launch chains:
#
# A. `srun --mpi=pmix` does NOT work for OpenMPI applications on this cluster.
#    Root cause: SLURM 25.05.5 provides PMIx v4 only (`pmix_v4` plugin), but
#    system OMPI 4.1.8 was built `--with-pmix=/slurm-build/package/usr` against
#    PMIx 3.x — its `ext3x` client cannot negotiate with the v4 server.
#    Symptom: `OPAL ERROR: Unreachable in ext3x_client.c at line 111` and
#    `OMPI was not built with SLURM's PMI support` at `MPI_Init_thread()`.
#    `--mpi=pmi2` fails the same way for OMPI apps.
#    Bare-metal `srun --mpi=pmix hostname` (probe #4) still works — that is
#    launcher-level PMIx, not OMPI's client.
#
# B. `mpirun -np N --host <list> --map-by node <python>` (Veros pattern) starts
#    ranks fine and `MPI_Init_thread()` succeeds, but the first inter-node
#    collective (`comm.gather`) hung until SLURM cancelled the job. Root cause
#    almost certainly TCP wireup auto-detecting the wrong interface (no UCX,
#    no libfabric, no IB verbs in the system OMPI MCA stack — only TCP/SM).
#    Layer A v2 retests with explicit `--mca btl tcp,self,vader` plus
#    `--mca btl_tcp_if_include <iface>` / `--mca oob_tcp_if_include <iface>`.
#
# Implication for Phase 7 launcher chain (revises Phase −1 finding #2):
#   Plan's "preferred" pyxis + `srun --mpi=pmix` chain is closed off by (A).
#   We will use the Veros chain regardless of pyxis availability:
#     - one container per node launched via plain `srun` (or pyxis, equivalent),
#       one task per node
#     - rank 0's container runs `mpirun --host <list> --map-by node ...` to
#       fan out via ssh into other containers (sshd in container required)
#   Confirmed by Layer A green run (job 1977112): `mpirun --host <list>` plus
#   `--mca btl tcp,self,vader --mca pml ob1 --mca btl_tcp_if_include <iface>`
#   passes the full mpi4py smoke (gather, bcast, Reduce float32, Reduce
#   complex64). `<iface>` is the head's default-route interface (varies by
#   node family — `enp226s0f0` on g492-*, `eno1np0` on ipp1-* — auto-detect
#   from `ip -4 -o route show default | awk '{print $5}'` works portably).
#   Exclude-by-name (`--mca btl_tcp_if_exclude`) is NOT sufficient: some node
#   generations have additional bridges (cni*, br-*, k8s_*) that we can't
#   enumerate ahead of time. Use include-by-name only.
#
# 6. mpi4py 4.1.1 (PyPI sdist) builds cleanly with `MPICC=/bin/mpicc` against
#    system OMPI 4.1.8 (Ubuntu 24.04 compute nodes). Build cached at
#    /home/dleshchev/.cache/pip/wheels/.../mpi4py-4.1.1-cp311-cp311-linux_x86_64.whl
#    NOTE: pixi env was originally provisioned inside Docker (/workspace), so
#    `bin/pip` has a stale shebang. Use `<pixi_env>/bin/python -m pip` instead.
#
# 7. **MPI.COMPLEX and MPI.C_FLOAT_COMPLEX are both available** in this
#    mpi4py + OMPI 4.1.8 build. Phase 3 (`distributed_mean`) can issue
#    `MPI.Reduce(...,MPI.C_FLOAT_COMPLEX,op=MPI.SUM)` on `np.complex64` buffers
#    directly — the float32×2 fallback path described in the plan is not needed
#    on this cluster. Reduce correctness verified by Layer A smoke.
#
# 8. **Bare-metal JAX-GPU works in the pixi env** (job 1977152 on ipp1-2165)
#    and bare-metal mpi4py reduce passes (job 1977183) — useful as a sanity
#    floor that confirms the cluster's mechanics. **Production path stays
#    inside Docker** for dependency pinning (CUDA toolkit, nsys, exact pixi env
#    snapshot). Phase 7's launcher will use the existing one-container-per-node
#    pattern with sshd-in-container so `mpirun --mca plm_rsh_agent "ssh -p 2222"`
#    can fan out across nodes inside the Docker network namespace.
#
# 9. **Docker MPI install works** (Layer C, job 1977983):
#    - Dockerfile adds `apt install openmpi-bin libopenmpi-dev openssh-server
#      openssh-client iproute2`; image rebuilds in ~17s.
#    - Inside the rebuilt image: `/usr/bin/mpirun` is OpenMPI 4.1.2 (Ubuntu
#      22.04 apt — 0.6 patch versions behind the cluster's bare-metal 4.1.8,
#      ABI-compatible at the OMPI 4.1.x level).
#    - The pixi-env mpi4py wheel (built bare-metal against 4.1.8) imports and
#      runs cleanly in the container against 4.1.2 (`Get_library_version()`
#      reports the in-container library, not the build-time one).
#    - 2-rank single-node `mpirun --allow-run-as-root --oversubscribe` inside
#      the container passes gather + Reduce.
# =============================================================================

set -e

STARTUP_T0=$(date +%s%3N)
echo "[$(date +%T.%3N)] Task ${SLURM_PROCID:-0} of ${SLURM_NTASKS:-1} on $(hostname) — startup begin"

CONTAINER_IMAGE="recovar:latest"
TARBALL_PATH="${SCRIPT_DIR}/recovar_container.tar"

# Load container from tarball (shared filesystem) if not already present
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^recovar:latest$"; then
    if [ -f "${TARBALL_PATH}" ]; then
        echo "[$(date +%T.%3N)] Loading container from tarball on $(hostname)..."
        docker load -i "${TARBALL_PATH}"
        echo "[$(date +%T.%3N)] Container loaded on $(hostname)"
    else
        echo "[$(date +%T.%3N)] No tarball found. Building container on $(hostname)..."
        bash "${SCRIPT_DIR}/scripts/build_container.sh"
        echo "[$(date +%T.%3N)] Container built on $(hostname)"
    fi
else
    echo "[$(date +%T.%3N)] Container image already present on $(hostname)"
fi
CONTAINER_T=$(date +%s%3N)
echo "[$(date +%T.%3N)] Container ready ($(( (CONTAINER_T - STARTUP_T0) ))ms)"

# Rank 0 installs pixi env first (shared filesystem — can't install concurrently).
# Other ranks wait for a marker file before running.
INSTALL_MARKER="${SCRIPT_DIR}/.pixi_install_done_${SLURM_JOB_ID}"

INSTALL_T0=$(date +%s%3N)
if [ "${SLURM_PROCID}" = "0" ]; then
    echo "[$(date +%T.%3N)] Rank 0: Installing pixi env (other ranks will wait)..."
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

            echo 'Cleaning stale editable installs...'
            rm -f /workspace/.pixi/envs/default/lib/python3.11/site-packages/__editable__*recovar* 2>/dev/null || true
            rm -f /workspace/.pixi/envs/default/lib/python3.11/site-packages/recovar.egg-link 2>/dev/null || true

            echo 'Installing RECOVAR...'
            pixi run install-recovar
            echo 'Install complete.'
        "
    touch "${INSTALL_MARKER}"
    sync  # Flush to NFS
    # Also force NFS cache invalidation by stat-ing the file
    stat "${INSTALL_MARKER}" > /dev/null 2>&1
    INSTALL_T1=$(date +%s%3N)
    echo "[$(date +%T.%3N)] Rank 0: Install marker written (install took $(( (INSTALL_T1 - INSTALL_T0) ))ms)"
else
    echo "[$(date +%T.%3N)] Rank ${SLURM_PROCID}: Waiting for rank 0 to finish install..."
    for i in $(seq 1 600); do
        # Force NFS cache invalidation by listing the directory
        ls "$(dirname "${INSTALL_MARKER}")" > /dev/null 2>&1
        if stat "${INSTALL_MARKER}" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if ! stat "${INSTALL_MARKER}" > /dev/null 2>&1; then
        echo "ERROR: Timed out waiting for rank 0 install (600s)"
        exit 1
    fi
    INSTALL_T1=$(date +%s%3N)
    echo "[$(date +%T.%3N)] Rank ${SLURM_PROCID}: Install marker found (waited $(( (INSTALL_T1 - INSTALL_T0) ))ms)"
fi

TASK_T0=$(date +%s%3N)
echo "[$(date +%T.%3N)] Rank ${SLURM_PROCID}: Starting task (total startup: $(( (TASK_T0 - STARTUP_T0) ))ms)"

# When running multiple tasks per node, restrict each container to one GPU.
DOCKER_GPU_FLAG="--runtime=nvidia"
if [ -n "${SLURM_LOCALID}" ] && [ "${SLURM_NTASKS_PER_NODE:-1}" -gt 1 ]; then
    DOCKER_GPU_FLAG="--gpus device=${SLURM_LOCALID}"
    echo "[$(date +%T.%3N)] Rank ${SLURM_PROCID}: pinning to GPU ${SLURM_LOCALID}"
fi

# Now run the actual task (pixi env already installed on shared filesystem)
docker run --rm --net host --ipc=host \
    ${DOCKER_GPU_FLAG} \
    -v "${SCRIPT_DIR}":/workspace \
    -w /workspace \
    --user "$(id -u):$(id -g)" \
    -e "SLURM_PROCID=${SLURM_PROCID}" \
    -e "SLURM_NTASKS=${SLURM_NTASKS}" \
    -e "SLURM_JOB_ID=${SLURM_JOB_ID}" \
    -e "SLURM_NODELIST=${SLURM_NODELIST}" \
    -e "KMP_DUPLICATE_LIB_OK=TRUE" \
    "${CONTAINER_IMAGE}" \
    -c "
        set -e
        echo \"Rank \${SLURM_PROCID}: Running task: ${TASK_CMD}\"
        ${TASK_CMD}
        echo \"Rank \${SLURM_PROCID}: Task completed!\"
    "

TASK_T1=$(date +%s%3N)
echo "[$(date +%T.%3N)] Rank ${SLURM_PROCID}: Task finished (task: $(( (TASK_T1 - TASK_T0) ))ms, total: $(( (TASK_T1 - STARTUP_T0) ))ms)"

# Note: marker cleanup is done in the batch script after srun completes,
# not here, to avoid removing it before rank 1 sees it.
