# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RECOVAR** is a scientific Python package for analyzing conformational heterogeneity in cryo-EM and cryo-ET datasets. It reconstructs 3D volumes, estimates conformational density, and detects low free-energy motions using PCA-based latent space methods.

- **GPU framework:** JAX with CUDA 12.6
- **Build/env manager:** Pixi (conda-based)
- **Entry point:** `recovar <command>`
- **Authors:** Marc Aurele Gilles (Princeton)
- **GitHub:** https://github.com/ma-gilles/recovar

## Common Commands

### Environment Setup
```bash
pixi install          # Install all dependencies
pixi shell            # Activate environment
```

### Creating Test Datasets
```bash
pixi run create-dataset-small    # 128x128, 100k images (~30 sec)
pixi run create-dataset-large    # 256x256, 300k images (~3 min)
```

### Running the Pipeline
```bash
pixi run pipeline-small          # Single GPU, 128x100k dataset
pixi run pipeline-large          # Single GPU, 256x300k dataset
pixi run pipeline-small-lazy     # Single GPU, lazy loading (saves memory)
pixi run pipeline-large-lazy     # Single GPU, lazy loading (saves memory)
```

### Full Workflow (create dataset + pipeline)
```bash
pixi run full-workflow-small         # 128x100k, standard loading
pixi run full-workflow-large         # 256x300k, standard loading
pixi run full-workflow-small-lazy    # 128x100k, lazy loading
pixi run full-workflow-large-lazy    # 256x300k, lazy loading
```

### Multi-GPU Testing
```bash
pixi run test-1gpu    # Single GPU baseline
pixi run test-2gpu    # 2 GPUs
pixi run test-4gpu    # 4 GPUs
pixi run test-8gpu    # 8 GPUs
```

### Profiling
```bash
# Focused on compute_H_B domain (cleaner profiles)
pixi run profile-pipeline-small          # 128x100k
pixi run profile-pipeline-large          # 256x300k

# Full capture (all NVTX domains)
pixi run profile-pipeline-small-full
pixi run profile-pipeline-large-full

# Exclude compute_H_B (analyze everything else)
pixi run profile-pipeline-small-no-hb
pixi run profile-pipeline-large-no-hb

# With data I/O annotations
pixi run profile-pipeline-small-with-io
pixi run profile-pipeline-large-with-io

# Multi-GPU profiling (128x100k with 50k images)
pixi run profile-1gpu
pixi run profile-2gpu
pixi run profile-4gpu

# Multi-GPU profiling (256x300k)
pixi run profile-1gpu-256
pixi run profile-2gpu-256
pixi run profile-4gpu-256
```

All pixi tasks are defined in `pixi.toml`. See there for the full list and exact commands.

## Job Submission Workflow

**You cannot run GPU workloads directly.** All tests, profiling, and pipeline runs must be submitted to the SLURM cluster via `sbatch`. Jobs can be submitted from login nodes or from within existing SLURM jobs.

### Submission Chain

```
./submit_job.sh <action>
  -> sbatch (SLURM submission, --gpus-per-node=N, --nodes=1)
    -> SLURM schedules job on x86_64 GPU node
      -> Batch script (generated in scripts/job_scripts/)
        -> Checks/builds Docker image (recovar:latest)
          -> docker run --runtime=nvidia, project mounted at /workspace
            -> pixi install && pixi run install-recovar
              -> pixi run <task>   (the actual workload)
```

### Key Components

| Component | Role |
|-----------|------|
| `submit_job.sh` | Top-level dispatcher. Maps action names to `submit_job()` calls with GPU count and time limit |
| `Dockerfile` | Based on `nvidia/cuda:12.6.0-devel-ubuntu22.04`. Includes Pixi and Nsight Systems 2025.5.1 |
| `scripts/build_container.sh` | Builds the Docker image if not already present on the node |
| `pixi.toml` [tasks] | Maps task names (e.g., `profile-2gpu`) to actual shell commands |

### How to Submit a Job

```bash
./submit_job.sh <action>
```

Run without arguments to see all available actions. Examples:
```bash
./submit_job.sh pipeline-small         # Single-GPU pipeline, 2h
./submit_job.sh test-2gpu              # 2-GPU test run, 30m
./submit_job.sh profile-2gpu           # 2-GPU profiling, 128-100k dataset, 1.5h
./submit_job.sh create-small           # Create 128-100k dataset, 1h
./submit_job.sh smoke-gpu              # Docker + GPU discovery + JAX import, 15m
```

### Docker GPU Visibility

The cluster's NVIDIA Docker runtime auto-discovers SLURM-allocated GPUs -- no `--gpus` flag is needed. The `--runtime=nvidia` flag is sufficient. Inside the container, `jax.devices("gpu")` will see exactly the GPUs allocated by SLURM.

### ARM Node Exclusion

Some nodes with A100 GPUs run ARM (aarch64) CPUs. The pixi environment only supports `linux-64` (x86_64). The script excludes known ARM nodes via `--exclude`. If a job fails with `unsupported-platform linux-aarch64`, add the node to `SLURM_EXCLUDE` in `submit_job.sh`.

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Watch job output (job ID shown after submission)
tail -f scripts/output/slurm-<JOB_ID>.out

# Move stray output files to scripts/output/
./submit_job.sh organize-outputs
```

## Architecture

### Pipeline Flow

```
Input (STAR/MRCS/CryoSPARC CS)
  -> Dataset Loading (supports lazy loading for large datasets)
  -> Preprocessing (CTF correction, masking, normalization)
  -> Covariance Estimation (hottest path)
  -> PCA/SVD (principal_components.py)
  -> Embedding (latent space assignment per particle)
  -> Heterogeneous Volume Reconstruction
  -> Output (volumes, embeddings, metrics)
```

### Command Architecture

- `recovar/command_line.py` -- CLI entry point; auto-discovers all `.py` files in `recovar/commands/`
- Each command module must define a `main()` function and is invoked as `recovar <module_name>`
- `recovar/commands/pipeline.py` -- the primary end-to-end command

### Core Modules

| Module | Purpose |
|--------|---------|
| `covariance_estimation.py` | Covariance matrix computation; performance-critical |
| `principal_components.py` | PCA/SVD on covariance |
| `embedding.py` | Latent space assignment |
| `homogeneous.py` | Mean volume reconstruction |
| `core.py` | Low-level cryo-EM ops (CTF, Fourier slicing) |
| `noise.py` | Noise model estimation |
| `simulator.py` | Synthetic dataset generation |
| `dataset.py` / `cryo_dataset.py` | Dataset management and I/O |
| `multi_gpu_utils.py` | Multi-GPU coordination |

### JAX & GPU Notes

- Computation is primarily JAX with `@jax.jit`-compiled kernels
- NVTX profiling annotations are used in key modules for Nsight Systems profiling
- Batch sizes are dynamically computed from available GPU memory (`utils.py`)
- Multi-GPU uses image-level parallelism

### Data Formats

Supports: RELION STAR, CryoSPARC CS, MRC/MRCS, plain text particle lists, and cryoDRGN outputs. Compatibility layers exist in `starfile.py`, `cryodrgn_load.py`, and `image_loader.py`.

## Multi-Node Distributed Pipeline (branch: multinode-dev)

### Status: Core implementation complete, I/O bottleneck identified

The pipeline has been refactored into discrete stage functions (`recovar/stages.py`) and a distributed command (`recovar pipeline_distributed`) added for multi-node execution via file-based coordination on shared filesystem.

### New Modules

| Module | Purpose |
|--------|---------|
| `recovar/stages.py` | Pipeline decomposed into 8 stage functions extracted from the monolithic `standard_recovar_pipeline` |
| `recovar/distributed.py` | Multi-node coordination: barriers, partial I/O, assignments, broadcast |
| `recovar/stage_checkpoint.py` | Per-stage checkpoint save/load with DONE markers |
| `recovar/distributed_stages.py` | Distributed wrappers for each parallel stage (image-split mean, freq-split covariance) |
| `recovar/commands/pipeline_distributed.py` | `recovar pipeline_distributed` entry point |

### Running Distributed Tests

```bash
# Smoke test: verify Docker + SLURM env on 2 nodes
./submit_job.sh smoke-multinode

# Generate reference checkpoints (1 node, ~50 min)
./submit_job.sh test-stage-ref

# Test individual stages with N ranks
./submit_job.sh test-stage-mean-1     # 1 node
./submit_job.sh test-stage-mean-2     # 2 nodes
./submit_job.sh test-stage-cov-1      # covariance, 1 node
./submit_job.sh test-stage-cov-2      # covariance, 2 nodes
./submit_job.sh test-stage-cov-4      # covariance, 4 ranks on 1 node

# Profiling distributed stages (nsys + NVTX)
./submit_job.sh profile-stage-cov-1   # 1 node
./submit_job.sh profile-stage-cov-2   # 2 nodes

# Compare outputs across node counts
./submit_job.sh test-stage-compare

# Full distributed pipeline
./submit_job.sh dist-small-1node      # single-node verification
./submit_job.sh dist-small-2node      # 2 nodes
./submit_job.sh dist-small-4node      # 4 nodes
```

### Test Results (128x128, 100k images, A100-80GB)

**Correctness (1-node vs 2-node, tolerance 1e-3):** All outputs PASS (mean, covariance H/B).

**Covariance stage profiling (detailed breakdown):**

| Config | Compute | Write partials | Barrier wait | Read/assemble | Total |
|--------|---------|---------------|-------------|---------------|-------|
| 1-rank | 570s | — | — | — | **570s** |
| 2-rank | 313s | 72s (rank 1) | 121s (rank 0) | 102s | **536s** |
| 4-rank | 196s | 240s (ranks 1-3) | 246s (rank 0) | 666s | **1108s** |

**Key finding:** Compute scales perfectly (570→313→196s) but NFS I/O for partials (2.5-5 GB each) dominates at higher rank counts. The 128-box dataset is too small to benefit — I/O overhead exceeds compute savings. The 256-box dataset (8x more compute, similar I/O) should show real speedup.

**Optimizations applied:**
- Rank 0 keeps its data in memory (no write-then-re-read of own partials)
- Non-rank-0 nodes return immediately after writing (no second barrier, no result loading)
- Assembly uses memmap-backed output arrays (disk-backed, low RAM usage)
- Test saves use numpy arrays instead of 20GB pickle

### Known Issues

1. **NFS I/O is the primary bottleneck for multi-rank covariance.** Writing/reading 2.5-5 GB partial arrays through NFS takes 20-125s each (variable throughput). For 128-box, I/O overhead exceeds compute savings at 4+ ranks. Need to test 256-box where compute dominates.

2. **Assembly OOM on low-RAM nodes.** A100-PCIe-40GB nodes have only 32-64 GB system RAM — too little to hold assembled H/B arrays (20 GB for 128-box, ~160 GB for 256-box). Mitigated with memmap-backed assembly (`_assemble_half_to_memmap`), but 32GB nodes still fail. Use high-RAM nodes (128GB+) or the single-node multi-rank mode (`submit_singlenode_multirank_job`).

3. **OpenMP duplicate library crash on multi-rank-per-node.** Multiple Docker containers on one node can hit conflicting OpenMP runtimes. Workaround: `KMP_DUPLICATE_LIB_OK=TRUE` (set in `run_node_container.sh`).

4. **Mean stage slower at multi-node.** Rank 0 post-processing (~140s) dominates the ~5s parallel accumulation.

5. **Mean and covariance+PCA are distributed.** Noise and embedding still run on rank 0 only.

6. **Docker install race.** Concurrent SLURM jobs sharing `.pixi` env can conflict. Rank 0 installs first, others wait via marker file.

7. **NFS barrier reliability.** File-based barrier uses `os.stat()` polling. Works but NFS attribute caching can cause delays. Timeout is 1800s.

8. **Tilt series not supported in distributed mode.** Image splitting must respect tilt-series boundaries. Deferred.

9. **Final `pipeline_complete` barrier + cleanup race.** Rank 0 cleans up `checkpoint_dir` after the final barrier, but rank 1 may still be polling for rank 0's barrier marker and never see it (cleanup removes it). Pipeline output is correct but the job hangs until SLURM kills it. Workaround: use `--keep-checkpoints` or cancel the job after rank 0 reports completion.

### Multi-Node Architecture

Two submission modes:
- **Multi-node:** `submit_multinode_job` — N nodes, 1 rank per node, `--runtime=nvidia`
- **Single-node multi-rank:** `submit_singlenode_multirank_job` — 1 node, N ranks, `--gpus device=K` per rank (useful when cluster can't schedule N separate GPU nodes)

```
submit_job.sh <action>
  -> sbatch with --nodes=N, --ntasks-per-node=M
    -> SLURM allocates nodes
      -> Head node: builds Docker + saves tarball
        -> srun launches run_node_container.sh per rank
          -> Each rank: docker run with GPU pinning + SLURM env vars
            -> recovar pipeline_distributed (or test stage runner)
              -> Stage-by-stage execution with file-based barriers
```

### NVTX Profiling

Two NVTX domains for Nsight Systems profiling:
- `compute_H_B` — JAX compute kernels, GPU transfers, multi-GPU orchestration
- `distributed` — partial I/O, barriers, assembly, checkpoint save/load

Run `./submit_job.sh profile-stage-cov-{1,2}` for nsys profiles with both domains enabled.

### Parallelization Strategy

| Stage | Strategy | Notes |
|-------|----------|-------|
| Setup | Rank 0 only | Fast, broadcasts config |
| Mean | Image-split | Each rank accumulates partial ft_y/ft_ctf, rank 0 reduces + post-processes |
| Mask | Rank 0 only | Fast |
| Noise + Variance | Rank 0 only | Deferred |
| Covariance+PCA | Frequency-split H/B, rank-0 regularization | All ranks compute distributed H/B (2 halves × N/2 freq ranges) via `distributed_covariance_hb`. Rank 0 then runs regularization, SVD, and rescaling on the assembled H/B. H/B is computed once and reused across all focus masks (it depends on `dilated_volume_mask`, not per-mask `focus_mask`). |
| Embedding | Rank 0 only | Deferred |
| Save | Rank 0 only | |

### Covariance+PCA Decomposition

`compute_regularized_covariance_columns` in `covariance_estimation.py` has been decomposed so the H/B step is separable:
- `compute_both_H_B(cryos, means, dilated_volume_mask, picked_frequencies, ...)` — the compute kernel
- `regularize_covariance_columns(Hs, Bs, cryo, mean_prior, volume_mask, valid_idx, gpu_memory, options, picked_frequencies)` — pure regularization
- `regularize_covariance_columns_in_batch(Hs, Bs, ...)` — batches the regularization over frequency columns
- `compute_regularized_covariance_columns` = `compute_both_H_B` + `regularize_covariance_columns` (unchanged behavior)

`principal_components.pick_covariance_frequencies(cryos, means, covariance_options, variance_estimate)` — extracted from `estimate_principal_components` so distributed code can pick frequencies independently.

`distributed_covariance_pca` (in `distributed_stages.py`) for `world_size > 1`:
1. All ranks: set up `covariance_options`, pick frequencies (cheap, deterministic)
2. All ranks: call `distributed_covariance_hb` to compute H/B across nodes
3. Non-rank-0: wait at barrier, load result
4. Rank 0: loop over focus masks → `regularize_covariance_columns_in_batch` → SVD → rescaling → contrast correction

### Checkpoint Barrier Race Fix

Previously, distributed stages called `checkpoint.mark_complete()` BEFORE the barrier. If rank 0 finished a stage fast, rank 1 could see the DONE marker via `ckpt.is_complete()` in the pipeline orchestrator, skip the stage (and its barrier), and desync. All distributed stage wrappers (`distributed_mean`, `distributed_mask`, `distributed_noise_refine_and_variance`, `distributed_covariance_pca`, `distributed_embedding`) now call `mark_complete()` AFTER the barrier.
