#!/bin/bash
# Run one Docker container per SLURM node for the RECOVAR_MPI=1 multi-node
# path. Each container starts an OpenSSH sshd on a per-job port; rank 0 then
# fans out to the other ranks via `mpirun --mca plm_rsh_agent "ssh -p <port>
# -i <key>"`. Worker ranks just run sshd and wait for rank 0 to finish.
#
# Why this pattern? Phase 0 probes (see scripts/run_node_container.sh comment
# header) found:
#  - `srun --mpi=pmix` is broken on this cluster: SLURM provides PMIx v4 but
#    the apt OpenMPI's ext3x client targets PMIx 3 → ABI mismatch on
#    MPI_Init_thread.
#  - The cluster authenticates SSH via GSSAPI/Kerberos and the user has no
#    ssh keypair — but containers cannot easily forward Kerberos creds, so
#    we sidestep with an ephemeral per-job ed25519 keypair (NFS-shared,
#    served via sshd_config's AuthorizedKeysFile, deleted by the batch
#    script after srun returns).
#  - Cross-node mpirun must filter to the head's default-route interface
#    (`--mca btl_tcp_if_include`) — auto-detection picks docker0 / cni* on
#    some node generations.
#
# Expected env vars (passed via srun --export=ALL):
#   SCRIPT_DIR              project root on host (= /workspace inside container)
#   COORD_DIR               host path of per-job coordination dir
#   CCOORD_DIR              same dir as seen inside container (/workspace/scripts/output/...)
#   SSHD_PORT               per-job port (30000 + JOBID%30000)
#   PRIMARY_IF              cluster Ethernet iface name (head's default-route iface)
#   HOSTS                   comma-separated host list for mpirun --host
#   PY_ARGS                 the python args, e.g. "scripts/smoke_mpi_gpu.py" or
#                           "-m recovar.commands.pipeline_distributed --foo bar"
#   EXTRA_ENV               extra "K=V K=V" exports forwarded into container
#   SLURM_PROCID, SLURM_NTASKS, SLURM_JOB_ID, SLURM_NODELIST  (set by srun)

set -e

CONTAINER_IMAGE="recovar:latest"
TARBALL_PATH="${SCRIPT_DIR}/recovar_container.tar"

# Refresh NFS attribute cache so we see the freshly-saved tarball, not the
# previous job's stale inode (otherwise docker load can hit "stale file
# handle" mid-read).
ls -la "${SCRIPT_DIR}" >/dev/null 2>&1 || true
ls -la "${TARBALL_PATH}" >/dev/null 2>&1 || true

if [ -f "${TARBALL_PATH}" ]; then
    # Always reload from tarball when present: a stale recovar:latest cached on
    # this node from an older Dockerfile (e.g. pre-openssh-server) would
    # silently fail at sshd launch otherwise. docker load is idempotent.
    # Copy to a local path first so the docker-load read isn't competing with
    # head-node concurrent NFS access on the same inode.
    LOCAL_TARBALL="/tmp/recovar_container_${SLURM_JOB_ID}.tar"
    echo "[$(date +%T)] $(hostname): copying tarball to local disk"
    cp "${TARBALL_PATH}" "${LOCAL_TARBALL}"
    echo "[$(date +%T)] $(hostname): loading container from local tarball"
    docker load -i "${LOCAL_TARBALL}"
    rm -f "${LOCAL_TARBALL}"
elif ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^recovar:latest$"; then
    echo "[$(date +%T)] $(hostname): no tarball, no cached image; building container"
    bash "${SCRIPT_DIR}/scripts/build_container.sh"
fi

# Rank-0 install marker (same convention as run_node_container.sh, so concurrent
# multi-rank-per-node jobs don't race on `pixi install`).
INSTALL_MARKER="${SCRIPT_DIR}/.pixi_install_done_${SLURM_JOB_ID}"
if [ "${SLURM_PROCID}" = "0" ]; then
    echo "[$(date +%T)] $(hostname) rank 0: installing pixi env"
    docker run --rm --net host --ipc=host \
        --runtime=nvidia \
        -v "${SCRIPT_DIR}":/workspace \
        -w /workspace \
        --user "$(id -u):$(id -g)" \
        "${CONTAINER_IMAGE}" \
        -c "
            set -e
            pixi install
            rm -f /workspace/.pixi/envs/default/lib/python3.11/site-packages/__editable__*recovar* 2>/dev/null || true
            rm -f /workspace/.pixi/envs/default/lib/python3.11/site-packages/recovar.egg-link 2>/dev/null || true
            pixi run install-recovar
        "
    touch "${INSTALL_MARKER}"
    sync
    stat "${INSTALL_MARKER}" >/dev/null 2>&1
    echo "[$(date +%T)] $(hostname) rank 0: install marker written"
else
    echo "[$(date +%T)] $(hostname) rank ${SLURM_PROCID}: waiting for rank 0 install"
    for i in $(seq 1 600); do
        ls "$(dirname "${INSTALL_MARKER}")" >/dev/null 2>&1
        [ -f "${INSTALL_MARKER}" ] && break
        sleep 1
    done
    if [ ! -f "${INSTALL_MARKER}" ]; then
        echo "ERROR: rank ${SLURM_PROCID} timed out waiting for rank 0 install"
        exit 1
    fi
fi

# Build the -e args for extra env (split on whitespace, prepend -e).
DOCKER_EXTRA_ENV=()
if [ -n "${EXTRA_ENV:-}" ]; then
    for kv in ${EXTRA_ENV}; do
        DOCKER_EXTRA_ENV+=(-e "$kv")
    done
fi

# Container script (heredoc with single quotes — vars are expanded inside the
# container at runtime, after `-e` injects them).
CONTAINER_SCRIPT='
set -e

# Per-rank sshd in /tmp (no root; per-job port avoids collisions across
# concurrent jobs, AuthorizedKeysFile points at the per-job pub key).
SSHD_DIR=$(mktemp -d -p /tmp sshd-XXXX)
HOST_KEY="$SSHD_DIR/ssh_host_ed25519_key"
PID_FILE="$SSHD_DIR/sshd.pid"
LOG_FILE="$SSHD_DIR/sshd.log"
ssh-keygen -q -N "" -t ed25519 -f "$HOST_KEY" >/dev/null
cat > "$SSHD_DIR/sshd_config" <<EOF
Port $SSHD_PORT
HostKey $HOST_KEY
PidFile $PID_FILE
AuthorizedKeysFile $KEY_PUB
PasswordAuthentication no
PubkeyAuthentication yes
StrictModes no
UsePAM no
PermitUserEnvironment no
PrintMotd no
PrintLastLog no
LogLevel ERROR
SyslogFacility AUTHPRIV
ChallengeResponseAuthentication no
GSSAPIAuthentication no
KbdInteractiveAuthentication no
EOF
/usr/sbin/sshd -f "$SSHD_DIR/sshd_config" -E "$LOG_FILE"
sleep 1
echo "[$(hostname) rank=$SLURM_PROCID] sshd PID=$(cat $PID_FILE 2>/dev/null) listening on $SSHD_PORT"

# Self-test ssh on the loopback before publishing READY (catches bad config
# fast — if THIS rank cannot ssh to itself, mpirun-from-rank-0 will fail too).
if ! ssh -p "$SSHD_PORT" -i "$KEY_PRIV" \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=5 \
        "$(hostname)" "true" 2>>"$LOG_FILE" ; then
    echo "[$(hostname) rank=$SLURM_PROCID] ssh self-test FAILED"
    cat "$LOG_FILE" || true
    kill $(cat $PID_FILE 2>/dev/null) 2>/dev/null || true
    exit 1
fi

touch "$READY_FILE"

if [ "$SLURM_PROCID" = "0" ]; then
    echo "[rank 0] waiting for all ranks ready..."
    # Other ranks gate on rank 0'\''s pixi-install marker before they even
    # start their container — NFS attribute caching can add ~100s of slack
    # between rank 0 finishing install and rank 1 noticing. 600s upper bound
    # covers cold container load + pixi install + NFS lag without false fail.
    for i in $(seq 1 $((SLURM_NTASKS-1))); do
        for j in $(seq 1 600); do
            ls "$COORD_DIR" >/dev/null 2>&1 || true
            [ -f "$COORD_DIR/ready-$i" ] && break
            sleep 1
        done
        if [ ! -f "$COORD_DIR/ready-$i" ]; then
            echo "[rank 0] FAIL: rank $i never became ready (10 min timeout)"
            kill $(cat $PID_FILE 2>/dev/null) 2>/dev/null || true
            exit 1
        fi
    done
    echo "[rank 0] all ranks ready; running mpirun"

    PIXI_PY=/workspace/.pixi/envs/default/bin/python
    SSH_AGENT="ssh -p $SSHD_PORT -i $KEY_PRIV -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes"

    # Forward env vars from EXTRA_ENV to the worker ranks. mpirun does NOT
    # propagate the rank-0 process env to remote ranks; sshd strips most
    # environment, so we have to explicitly pass each KEY=VALUE via `-x`.
    # Format: EXTRA_ENV is a space-separated "K=V K=V ..." string.
    MPI_X_FLAGS=""
    if [ -n "${EXTRA_ENV:-}" ]; then
        for kv in ${EXTRA_ENV}; do
            MPI_X_FLAGS="$MPI_X_FLAGS -x ${kv%%=*}"
        done
    fi
    # Also forward a few common Recovar/JAX/CUDA env vars that the user may
    # rely on (use the rank-0 container value; note --net host).
    for v in PATH LD_LIBRARY_PATH HOME PYTHONPATH XLA_FLAGS \
             KMP_DUPLICATE_LIB_OK PYTHONUNBUFFERED; do
        if env | grep -q "^${v}="; then
            MPI_X_FLAGS="$MPI_X_FLAGS -x $v"
        fi
    done
    echo "[rank 0] mpirun -x flags: $MPI_X_FLAGS"

    set +e
    mpirun \
        -np "$SLURM_NTASKS" \
        --host "$HOSTS" \
        --map-by node \
        --bind-to none \
        --mca plm_rsh_agent "$SSH_AGENT" \
        --mca btl tcp,self,vader \
        --mca pml ob1 \
        --mca btl_tcp_if_include "$PRIMARY_IF" \
        --mca oob_tcp_if_include "$PRIMARY_IF" \
        $MPI_X_FLAGS \
        "$PIXI_PY" $PY_ARGS
    rc=$?
    set -e
    echo "[rank 0] mpirun exit=$rc"
    touch "$DONE0_FILE"
    kill $(cat $PID_FILE 2>/dev/null) 2>/dev/null || true
    exit $rc
else
    echo "[rank $SLURM_PROCID] sshd up; waiting for rank 0 to finish"
    for j in $(seq 1 7200); do
        ls "$COORD_DIR" >/dev/null 2>&1 || true
        [ -f "$DONE0_FILE" ] && break
        sleep 2
    done
    kill $(cat $PID_FILE 2>/dev/null) 2>/dev/null || true
    echo "[rank $SLURM_PROCID] exiting"
fi
'

# When running multiple ranks per node (single-node-multirank), restrict each
# container to one GPU (mirrors run_node_container.sh).
DOCKER_GPU_FLAG="--runtime=nvidia"
if [ -n "${SLURM_LOCALID}" ] && [ "${SLURM_NTASKS_PER_NODE:-1}" -gt 1 ]; then
    DOCKER_GPU_FLAG="--gpus device=${SLURM_LOCALID}"
fi

docker run --rm --net host --ipc=host \
    ${DOCKER_GPU_FLAG} \
    -v "${SCRIPT_DIR}":/workspace \
    -v "${HOME}":"${HOME}" \
    -e "HOME=${HOME}" \
    -e "SLURM_PROCID=${SLURM_PROCID}" \
    -e "SLURM_NTASKS=${SLURM_NTASKS}" \
    -e "SLURM_JOB_ID=${SLURM_JOB_ID}" \
    -e "SLURM_NODELIST=${SLURM_NODELIST}" \
    -e "PRIMARY_IF=${PRIMARY_IF}" \
    -e "HOSTS=${HOSTS}" \
    -e "COORD_DIR=${CCOORD_DIR}" \
    -e "READY_FILE=${CCOORD_DIR}/ready-${SLURM_PROCID}" \
    -e "DONE0_FILE=${CCOORD_DIR}/done-0" \
    -e "SSHD_PORT=${SSHD_PORT}" \
    -e "KEY_PRIV=${CCOORD_DIR}/job_key" \
    -e "KEY_PUB=${CCOORD_DIR}/job_key.pub" \
    -e "PY_ARGS=${PY_ARGS}" \
    -e "KMP_DUPLICATE_LIB_OK=TRUE" \
    -e "EXTRA_ENV=${EXTRA_ENV:-}" \
    "${DOCKER_EXTRA_ENV[@]}" \
    -w /workspace \
    --user "$(id -u):$(id -g)" \
    "${CONTAINER_IMAGE}" -c "${CONTAINER_SCRIPT}"
