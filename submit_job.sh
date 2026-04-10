#!/bin/bash

# Submit RECOVAR jobs to the SLURM cluster
# Usage: ./submit_job.sh <action>
#
# Jobs run inside a Docker container (recovar:latest) that:
#   1. Mounts this project at /workspace
#   2. Installs pixi env + RECOVAR
#   3. Executes a pixi task
#
# All pixi tasks are defined in pixi.toml.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPTS_DIR="$SCRIPT_DIR/scripts/job_scripts"
OUTPUT_DIR="$SCRIPT_DIR/scripts/output"

# Exclude ARM (aarch64) nodes — pixi env only supports linux-64 (x86_64).
# All mp32ar200 chassis nodes are ARM (Ampere Altra), even the g242-p3x ones
# without the "altra" prefix.
SLURM_EXCLUDE="${SLURM_EXCLUDE:-altra-g242-p32-02,altra-g242-p32-03,altra-g242-p32-04,altra-g242-p32-05,altra-g242-p33-01,g242-p33-0001,g242-p33-0002}"

# Create directories if they don't exist
mkdir -p "$JOB_SCRIPTS_DIR"
mkdir -p "$OUTPUT_DIR"

# Function to generate a batch script
generate_batch_script() {
    local task_cmd=$1
    local num_gpus=$2
    local time_limit=$3
    local task_name=$4
    local pid=$$
    local ts=$(date +%Y%m%d_%H%M%S)
    local batch_script="$JOB_SCRIPTS_DIR/recovar_${ts}_${pid}.sh"

    cat > "$batch_script" <<EOF
#!/bin/bash
#SBATCH --job-name=recovar-${task_name// /_}
#SBATCH --output=${OUTPUT_DIR}/slurm-%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm-%j.err
#SBATCH --gpus-per-node=${num_gpus}
#SBATCH --nodes=1
#SBATCH --time=${time_limit}
#SBATCH --exclude=${SLURM_EXCLUDE}
set -e

echo "=========================================="
echo "RECOVAR Batch Job Starting"
echo "Job Name: ${task_name}"
echo "Node: \$(hostname)"
echo "Date: \$(date)"
echo "GPUs: \$(nvidia-smi -L 2>/dev/null || echo 'nvidia-smi not available')"
echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=========================================="

# Configuration
SCRIPT_DIR="${SCRIPT_DIR}"
CONTAINER_IMAGE="recovar:latest"
TASK_CMD="${task_cmd}"

cd "\$SCRIPT_DIR"

# Check if container image exists
if ! docker images | grep -q "recovar.*latest"; then
    echo "Container image not found. Building..."
    bash scripts/build_container.sh
else
    echo "Container image found: \$CONTAINER_IMAGE"
fi

# GPU visibility: Docker on this cluster auto-discovers SLURM-allocated GPUs.
# No --gpus flag needed — the NVIDIA runtime handles it.

echo "Starting container and running task..."
echo "Task command: \$TASK_CMD"

docker run --rm --net host --ipc=host \\
    --runtime=nvidia \\
    -v "\$SCRIPT_DIR":/workspace \\
    -w /workspace \\
    --user $(id -u):$(id -g) \\
    "\$CONTAINER_IMAGE" \\
    -c "
        set -e
        echo 'Installing dependencies...'
        pixi install

        echo 'Installing RECOVAR...'
        pixi run install-recovar

        echo 'Running pixi task: \$TASK_CMD'
        \$TASK_CMD

        echo 'Task completed successfully!'
    "

echo "=========================================="
echo "RECOVAR Batch Job Completed"
echo "Date: \$(date)"
echo "=========================================="
EOF

    # Make the script executable
    chmod +x "$batch_script"

    echo "$batch_script"
}

# Function to generate a MULTI-NODE batch script
generate_multinode_batch_script() {
    local task_cmd=$1
    local num_gpus_per_node=$2
    local num_nodes=$3
    local time_limit=$4
    local task_name=$5
    local pid=$$
    local ts=$(date +%Y%m%d_%H%M%S)
    local batch_script="$JOB_SCRIPTS_DIR/recovar_mn_${ts}_${pid}.sh"

    # Escape task_cmd for safe embedding in script
    local escaped_task_cmd
    escaped_task_cmd=$(printf '%s' "$task_cmd" | sed "s/'/'\\\\''/g")

    cat > "$batch_script" <<BATCHEOF
#!/bin/bash
#SBATCH --job-name=recovar-${task_name// /_}
#SBATCH --output=${OUTPUT_DIR}/slurm-%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm-%j.err
#SBATCH --gpus-per-node=${num_gpus_per_node}
#SBATCH --nodes=${num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --time=${time_limit}
#SBATCH --exclude=${SLURM_EXCLUDE}
set -e

echo "=========================================="
echo "RECOVAR Multi-Node Batch Job Starting"
echo "Job Name: ${task_name}"
echo "Nodes: ${num_nodes}"
echo "GPUs per node: ${num_gpus_per_node}"
echo "Node: \$(hostname)"
echo "Date: \$(date)"
echo "SLURM_JOB_ID: \$SLURM_JOB_ID"
echo "SLURM_NODELIST: \$SLURM_NODELIST"
echo "=========================================="

export SCRIPT_DIR="${SCRIPT_DIR}"
export TASK_CMD='${escaped_task_cmd}'

cd "\$SCRIPT_DIR"

# Build container and save tarball on head node (shared filesystem).
# Worker nodes will load from tarball instead of rebuilding.
TARBALL_PATH="\$SCRIPT_DIR/recovar_container.tar"
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^recovar:latest\$'; then
    echo "Container image not found on head node. Building..."
    bash scripts/build_container.sh
fi
if [ ! -f "\$TARBALL_PATH" ]; then
    echo "Saving container tarball for worker nodes..."
    docker save recovar:latest -o "\$TARBALL_PATH"
    echo "Tarball saved"
else
    echo "Container tarball already exists: \$TARBALL_PATH"
fi

# Launch one container per node via srun.
# Each node runs scripts/run_node_container.sh which loads from tarball.
# SLURM_PROCID and SLURM_NTASKS are set by srun for each task.
srun --ntasks-per-node=1 --export=ALL bash "\$SCRIPT_DIR/scripts/run_node_container.sh"

echo "=========================================="
echo "RECOVAR Multi-Node Batch Job Completed"
echo "Date: \$(date)"
echo "=========================================="
BATCHEOF

    chmod +x "$batch_script"
    echo "$batch_script"
}

# Function to submit a multi-node job
submit_multinode_job() {
    local task_name=$1
    local task_cmd=$2
    local num_gpus_per_node=$3
    local num_nodes=$4
    local time_limit=$5

    echo "========================================"
    echo "Submitting multi-node job: $task_name"
    echo "Task command: $task_cmd"
    echo "GPUs per node: $num_gpus_per_node"
    echo "Number of nodes: $num_nodes"
    echo "Time limit: $time_limit"
    echo "========================================"

    batch_script=$(generate_multinode_batch_script "$task_cmd" "$num_gpus_per_node" "$num_nodes" "$time_limit" "$task_name")
    echo "Generated batch script: $batch_script"

    local sbatch_output=$(sbatch "$batch_script" 2>&1)
    echo "$sbatch_output"
    local job_id=$(echo "$sbatch_output" | grep -oP '\d+$')

    if [ -n "$job_id" ]; then
        echo ""
        echo "Job ID: $job_id"
        echo "Output file: ${OUTPUT_DIR}/slurm-${job_id}.out"
        echo ""
        echo "To monitor: tail -f ${OUTPUT_DIR}/slurm-${job_id}.out"
        echo "To check status: squeue -j $job_id"
    fi

    echo "Job submitted successfully!"
    echo ""
}

# Function to submit a job
submit_job() {
    local task_name=$1
    local task_cmd=$2
    local num_gpus=$3
    local time_limit=$4

    echo "========================================"
    echo "Submitting job: $task_name"
    echo "Task command: $task_cmd"
    echo "Number of GPUs: $num_gpus"
    echo "Time limit: $time_limit"
    echo "========================================"

    # Generate the batch script
    batch_script=$(generate_batch_script "$task_cmd" "$num_gpus" "$time_limit" "$task_name")
    echo "Generated batch script: $batch_script"

    local sbatch_output=$(sbatch "$batch_script" 2>&1)
    echo "$sbatch_output"
    local job_id=$(echo "$sbatch_output" | grep -oP '\d+$')

    if [ -n "$job_id" ]; then
        echo ""
        echo "Job ID: $job_id"
        echo "Output file: ${OUTPUT_DIR}/slurm-${job_id}.out"
        echo ""
        echo "To monitor: tail -f ${OUTPUT_DIR}/slurm-${job_id}.out"
        echo "To check status: squeue -j $job_id"
    fi

    echo "Job submitted successfully!"
    echo ""
}

# Main action handler
ACTION=$1

if [ -z "$ACTION" ]; then
    echo "Usage: $0 <action>"
    echo ""
    echo "Available actions:"
    echo ""
    echo "  Smoke tests:"
    echo "    smoke-gpu        - Docker build + GPU discovery + JAX import (15m)"
    echo "    smoke-import     - Verify pixi env + recovar import (15m)"
    echo "    smoke-test       - Run recovar's built-in test dataset (30m)"
    echo ""
    echo "  Pipeline runs (128-100k dataset):"
    echo "    pipeline-small        - 1 GPU pipeline (2h)"
    echo "    pipeline-small-lazy   - 1 GPU pipeline with lazy loading (2h)"
    echo ""
    echo "  Pipeline runs (256-300k dataset):"
    echo "    pipeline-large        - 1 GPU pipeline (2h)"
    echo "    pipeline-large-lazy   - 1 GPU pipeline with lazy loading (4h)"
    echo ""
    echo "  Full workflow (create dataset + pipeline):"
    echo "    full-workflow-small      - 128-100k, standard loading (2h)"
    echo "    full-workflow-large      - 256-300k, standard loading (4h)"
    echo "    full-workflow-small-lazy - 128-100k, lazy loading (2h)"
    echo "    full-workflow-large-lazy - 256-300k, lazy loading (4h)"
    echo ""
    echo "  Multi-GPU tests (128-100k dataset):"
    echo "    test-1gpu       - Test with 1 GPU (30m)"
    echo "    test-2gpu       - Test with 2 GPUs (30m)"
    echo "    test-4gpu       - Test with 4 GPUs (30m)"
    echo "    test-8gpu       - Test with 8 GPUs (30m)"
    echo ""
    echo "  Dataset creation:"
    echo "    create-small    - Create 128-100k dataset (1h)"
    echo "    create-large    - Create 256-300k dataset (2h)"
    echo ""
    echo "  Profiling (128-100k dataset):"
    echo "    profile-1gpu    - Profile 1 GPU run (2h)"
    echo "    profile-2gpu    - Profile 2 GPU run (1.5h)"
    echo "    profile-4gpu    - Profile 4 GPU run (45m)"
    echo ""
    echo "  Profiling (256-300k dataset):"
    echo "    profile-1gpu-256 - Profile 1 GPU run (3h)"
    echo "    profile-2gpu-256 - Profile 2 GPU run (2h)"
    echo "    profile-4gpu-256 - Profile 4 GPU run (1.5h)"
    echo ""
    echo "  Comparison:"
    echo "    compare-all     - Compare outputs from all multi-GPU runs (10m)"
    echo ""
    echo "  Distributed pipeline (multi-node):"
    echo "    dist-small-1node     - Distributed pipeline, 1 node, 1 GPU (verification) (2h)"
    echo "    dist-small-2node     - Distributed pipeline, 2 nodes, 1 GPU/node (2h)"
    echo "    dist-small-4node     - Distributed pipeline, 4 nodes, 1 GPU/node (2h)"
    echo "    dist-large-2node     - Distributed pipeline, 2 nodes, 2 GPU/node (2h)"
    echo "    dist-large-4node     - Distributed pipeline, 4 nodes, 2 GPU/node (2h)"
    echo "    smoke-multinode      - Multi-node smoke test: print rank/world_size (15m)"
    echo ""
    echo "  Stage-by-stage distributed tests:"
    echo "    test-stage-ref        - Generate reference checkpoints (1 node, 2h)"
    echo "    test-stage-mean-1     - Test mean stage, 1 node (30m)"
    echo "    test-stage-mean-2     - Test mean stage, 2 nodes (30m)"
    echo "    test-stage-mean-4     - Test mean stage, 4 nodes (30m)"
    echo "    test-stage-cov-1      - Test covariance H/B stage, 1 node (1h)"
    echo "    test-stage-cov-2      - Test covariance H/B stage, 2 nodes (1h)"
    echo "    test-stage-cov-4      - Test covariance H/B stage, 4 nodes (1h)"
    echo "    test-stage-compare    - Compare all stage test outputs (10m)"
    echo ""
    echo "  Utilities:"
    echo "    organize-outputs     - Move all slurm-*.out files to scripts/output/"
    echo ""
    exit 1
fi

case $ACTION in
    # Smoke tests
    smoke-gpu)
        submit_job "Smoke GPU" "pixi run smoke-gpu" 2 "00:15:00"
        ;;
    smoke-import)
        submit_job "Smoke Import" "pixi run smoke-import-recovar" 1 "00:15:00"
        ;;
    smoke-test)
        submit_job "Smoke Test" "pixi run test-recovar" 1 "00:30:00"
        ;;

    # Pipeline runs
    pipeline-small)
        submit_job "Pipeline Small" "pixi run pipeline-small" 1 "02:00:00"
        ;;
    pipeline-small-lazy)
        submit_job "Pipeline Small Lazy" "pixi run pipeline-small-lazy" 1 "02:00:00"
        ;;
    pipeline-large)
        submit_job "Pipeline Large" "pixi run pipeline-large" 1 "02:00:00"
        ;;
    pipeline-large-lazy)
        submit_job "Pipeline Large Lazy" "pixi run pipeline-large-lazy" 1 "04:00:00"
        ;;

    # Full workflows
    full-workflow-small)
        submit_job "Full Workflow Small" "pixi run full-workflow-small" 1 "02:00:00"
        ;;
    full-workflow-large)
        submit_job "Full Workflow Large" "pixi run full-workflow-large" 1 "04:00:00"
        ;;
    full-workflow-small-lazy)
        submit_job "Full Workflow Small Lazy" "pixi run full-workflow-small-lazy" 1 "02:00:00"
        ;;
    full-workflow-large-lazy)
        submit_job "Full Workflow Large Lazy" "pixi run full-workflow-large-lazy" 1 "04:00:00"
        ;;

    # Multi-GPU tests
    test-1gpu)
        submit_job "Test 1 GPU" "pixi run test-1gpu" 1 "00:30:00"
        ;;
    test-2gpu)
        submit_job "Test 2 GPUs" "pixi run test-2gpu" 2 "00:30:00"
        ;;
    test-4gpu)
        submit_job "Test 4 GPUs" "pixi run test-4gpu" 4 "00:30:00"
        ;;
    test-8gpu)
        submit_job "Test 8 GPUs" "pixi run test-8gpu" 8 "00:30:00"
        ;;

    # Dataset creation
    create-small)
        submit_job "Create Small Dataset" "pixi run create-dataset-small" 1 "01:00:00"
        ;;
    create-large)
        submit_job "Create Large Dataset" "pixi run create-dataset-large" 1 "02:00:00"
        ;;

    # Profiling (128-100k dataset)
    profile-1gpu)
        submit_job "Profile 1 GPU (128-100k)" "pixi run profile-1gpu" 1 "02:00:00"
        ;;
    profile-2gpu)
        submit_job "Profile 2 GPUs (128-100k)" "pixi run profile-2gpu" 2 "01:30:00"
        ;;
    profile-4gpu)
        submit_job "Profile 4 GPUs (128-100k)" "pixi run profile-4gpu" 4 "00:45:00"
        ;;

    # Profiling (256-300k dataset)
    profile-1gpu-256)
        submit_job "Profile 1 GPU (256-300k)" "pixi run profile-1gpu-256" 1 "03:00:00"
        ;;
    profile-2gpu-256)
        submit_job "Profile 2 GPUs (256-300k)" "pixi run profile-2gpu-256" 2 "02:00:00"
        ;;
    profile-4gpu-256)
        submit_job "Profile 4 GPUs (256-300k)" "pixi run profile-4gpu-256" 4 "01:30:00"
        ;;

    # Comparison
    compare-all)
        submit_job "Compare Multi-GPU Outputs" "pixi run compare-all-multigpu" 1 "00:10:00"
        ;;

    # Utilities
    organize-outputs)
        echo "========================================"
        echo "Organizing output files"
        echo "========================================"
        shopt -s nullglob
        files=("$SCRIPT_DIR"/slurm-*.out)
        if [ ${#files[@]} -eq 0 ]; then
            echo "No slurm output files found in root directory"
        else
            echo "Moving ${#files[@]} files to scripts/output/"
            for file in "${files[@]}"; do
                filename=$(basename "$file")
                echo "  Moving $filename"
                mv "$file" "$OUTPUT_DIR/"
            done
            echo "Done! All output files moved to scripts/output/"
        fi
        ;;

    # Distributed pipeline (single-node verification)
    dist-small-1node)
        submit_job "Dist Pipeline Small 1-Node" "pixi run pipeline-distributed-small" 1 "02:00:00"
        ;;

    # Distributed pipeline (multi-node)
    dist-small-2node)
        submit_multinode_job "Dist Pipeline Small 2-Node" "pixi run pipeline-distributed-small" 1 2 "02:00:00"
        ;;
    dist-small-4node)
        submit_multinode_job "Dist Pipeline Small 4-Node" "pixi run pipeline-distributed-small" 1 4 "02:00:00"
        ;;
    dist-large-2node)
        submit_multinode_job "Dist Pipeline Large 2-Node" "pixi run pipeline-distributed-large" 2 2 "02:00:00"
        ;;
    dist-large-4node)
        submit_multinode_job "Dist Pipeline Large 4-Node" "pixi run pipeline-distributed-large" 2 4 "02:00:00"
        ;;

    # Stage-by-stage distributed tests
    test-stage-ref)
        submit_job "Stage Test Reference" "pixi run test-stage-reference" 1 "02:00:00"
        ;;
    test-stage-mean-1)
        submit_job "Stage Test Mean 1-Node" "pixi run test-stage-mean" 1 "00:30:00"
        ;;
    test-stage-mean-2)
        submit_multinode_job "Stage Test Mean 2-Node" "pixi run test-stage-mean" 1 2 "00:30:00"
        ;;
    test-stage-mean-4)
        submit_multinode_job "Stage Test Mean 4-Node" "pixi run test-stage-mean" 1 4 "00:30:00"
        ;;
    test-stage-cov-1)
        submit_job "Stage Test Covariance 1-Node" "pixi run test-stage-covariance" 1 "01:00:00"
        ;;
    test-stage-cov-2)
        submit_multinode_job "Stage Test Covariance 2-Node" "pixi run test-stage-covariance" 1 2 "01:00:00"
        ;;
    test-stage-cov-4)
        submit_multinode_job "Stage Test Covariance 4-Node" "pixi run test-stage-covariance" 1 4 "01:00:00"
        ;;
    test-stage-compare)
        submit_job "Stage Test Compare" "pixi run compare-stage-mean && pixi run compare-stage-covariance" 1 "00:10:00"
        ;;

    # Multi-node smoke test
    smoke-multinode)
        submit_multinode_job "Smoke Multi-Node" "python3 scripts/smoke_multinode.py" 1 2 "00:15:00"
        ;;

    *)
        echo "Error: Unknown action '$ACTION'"
        echo "Run '$0' without arguments to see available actions."
        exit 1
        ;;
esac
