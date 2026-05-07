#!/bin/bash
# Run distributed stage tests inside a Docker container.
# Called from within a SLURM job.
#
# Usage:
#   run_distributed_stage_tests.sh reference 128
#   run_distributed_stage_tests.sh run-stage mean 128
#   run_distributed_stage_tests.sh compare mean 128 1 2

set -e

BASE_DIR="/workspace"
ACTION=$1
STAGE=$2
IMAGE_SIZE=${3:-128}
N_IMAGES=${4:-100000}

DATASET_DIR="${BASE_DIR}/data-${IMAGE_SIZE}-${N_IMAGES}/test_dataset"
REF_CHECKPOINT="${DATASET_DIR}/stage_test_reference_checkpoint"

case "$ACTION" in
    reference)
        echo "=== Phase 1: Generating reference checkpoints ==="
        python3 ${BASE_DIR}/scripts/test_stage_runner.py reference \
            --dataset-dir "$DATASET_DIR" \
            --image-size "$IMAGE_SIZE"
        ;;

    run-stage)
        WORLD_SIZE=${SLURM_NTASKS:-1}
        OUTPUT_DIR="${DATASET_DIR}/stage_test_${STAGE}_${WORLD_SIZE}node"
        echo "=== Phase 2: Running stage '${STAGE}' with ${WORLD_SIZE} node(s) ==="
        python3 ${BASE_DIR}/scripts/test_stage_runner.py run-stage \
            --stage "$STAGE" \
            --ref-checkpoint "$REF_CHECKPOINT" \
            --output-dir "$OUTPUT_DIR"
        ;;

    compare)
        REF_NODES=${5:-1}
        TEST_NODES=${6:-2}
        REF_OUT="${DATASET_DIR}/stage_test_${STAGE}_${REF_NODES}node"
        TEST_OUT="${DATASET_DIR}/stage_test_${STAGE}_${TEST_NODES}node"
        echo "=== Phase 3: Comparing ${STAGE}: ${REF_NODES} node vs ${TEST_NODES} nodes ==="
        python3 ${BASE_DIR}/scripts/test_stage_runner.py compare \
            --ref-dir "$REF_OUT" \
            --test-dir "$TEST_OUT" \
            --stage "$STAGE"
        ;;

    compare-ref)
        # Compare distributed 1-node output against reference pipeline checkpoint
        TEST_NODES=${5:-1}
        TEST_OUT="${DATASET_DIR}/stage_test_${STAGE}_${TEST_NODES}node"
        echo "=== Comparing ${STAGE}: reference pipeline vs ${TEST_NODES}-node distributed ==="
        python3 ${BASE_DIR}/scripts/test_stage_runner.py compare \
            --ref-dir "$REF_CHECKPOINT" \
            --test-dir "$TEST_OUT" \
            --stage "$STAGE"
        ;;

    profile-stage)
        WORLD_SIZE=${SLURM_NTASKS:-1}
        RANK=${SLURM_PROCID:-0}
        OUTPUT_DIR="${DATASET_DIR}/stage_profile_${STAGE}_${WORLD_SIZE}node"
        PROFILE_OUT="${BASE_DIR}/scripts/output/profile_${STAGE}_${WORLD_SIZE}node_rank${RANK}_$(date +%Y%m%d_%H%M%S)"
        echo "=== Profiling stage '${STAGE}' with ${WORLD_SIZE} node(s), rank ${RANK} ==="
        echo "Profile output: ${PROFILE_OUT}.nsys-rep"
        nsys profile \
            -t cuda,nvtx \
            -f true \
            -d 3600 \
            --nvtx-domain-include="compute_H_B,distributed" \
            -o "${PROFILE_OUT}" \
            python3 ${BASE_DIR}/scripts/test_stage_runner.py run-stage \
                --stage "$STAGE" \
                --ref-checkpoint "$REF_CHECKPOINT" \
                --output-dir "$OUTPUT_DIR"
        ;;

    *)
        echo "Usage: $0 {reference|run-stage|compare|compare-ref|profile-stage} <stage> [image_size] [n_images] [ref_nodes] [test_nodes]"
        echo ""
        echo "Stages: mean, noise_variance, covariance_hb, embedding"
        exit 1
        ;;
esac
