#!/bin/bash

# Run the distributed pipeline on test datasets.
# Usage:
#   ./run_test_distributed.sh pipeline 128 100000           # 1-node distributed (verification)
#   ./run_test_distributed.sh pipeline 128 100000 lazy      # With lazy loading
#   ./run_test_distributed.sh pipeline 256 300000           # Large dataset

set -e

BASE_DIR="/workspace"
ACTION=${1:-help}
IMAGE_SIZE=${2:-128}
N_IMAGES=${3:-100000}
LAZY_MODE=${4:-}

DATASET_DIR="${BASE_DIR}/data-${IMAGE_SIZE}-${N_IMAGES}"

run_distributed_pipeline() {
    echo "=========================================="
    echo "Running distributed recovar pipeline..."
    echo "Dataset directory: $DATASET_DIR"
    echo "Image size: $IMAGE_SIZE"
    echo "SLURM_PROCID: ${SLURM_PROCID:-0}"
    echo "SLURM_NTASKS: ${SLURM_NTASKS:-1}"
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
    echo "Lazy loading: $([ "$LAZY_MODE" == "lazy" ] && echo "ENABLED" || echo "DISABLED")"
    echo "=========================================="

    if [ ! -d "$DATASET_DIR/test_dataset" ]; then
        echo "Error: Dataset not found at $DATASET_DIR/test_dataset"
        echo "Please create the dataset first."
        exit 1
    fi

    cd "$DATASET_DIR/test_dataset/"

    PIPELINE_CMD="recovar pipeline_distributed particles.${IMAGE_SIZE}.mrcs --ctf ctf.pkl --poses poses.pkl --mask=from_halfmaps -o pipeline_distributed_output --checkpoint-dir ${DATASET_DIR}/test_dataset/checkpoint"

    if [ "$LAZY_MODE" == "lazy" ]; then
        PIPELINE_CMD="$PIPELINE_CMD --lazy"
    fi

    echo "Running: $PIPELINE_CMD"
    eval "$PIPELINE_CMD"

    echo "Distributed pipeline completed!"
    echo "Output at: $DATASET_DIR/test_dataset/pipeline_distributed_output"
}

case "$ACTION" in
    pipeline)
        run_distributed_pipeline
        ;;
    help|--help|-h)
        echo "Usage: $0 pipeline [image_size] [n_images] [lazy]"
        ;;
    *)
        echo "Error: Unknown action '$ACTION'"
        exit 1
        ;;
esac
