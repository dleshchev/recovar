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

### Status: Core implementation complete, stage-by-stage tests passing

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

# Test individual stages with N nodes
./submit_job.sh test-stage-mean-1     # 1 node
./submit_job.sh test-stage-mean-2     # 2 nodes
./submit_job.sh test-stage-mean-4     # 4 nodes
./submit_job.sh test-stage-cov-1      # covariance, 1 node
./submit_job.sh test-stage-cov-2      # covariance, 2 nodes
./submit_job.sh test-stage-cov-4      # covariance, 4 nodes

# Compare outputs across node counts
./submit_job.sh test-stage-compare

# Full distributed pipeline
./submit_job.sh dist-small-1node      # single-node verification
./submit_job.sh dist-small-2node      # 2 nodes
./submit_job.sh dist-small-4node      # 4 nodes
```

### Test Results (128x128, 100k images)

**Correctness (1-node vs 2-node, tolerance 1e-3):**

| Output | Relative Error | Status |
|--------|---------------|--------|
| Mean combined | 6.31e-08 | PASS |
| Mean corrected0 | 6.30e-08 | PASS |
| Mean corrected1 | 1.23e-07 | PASS |
| Mean prior | 9.45e-05 | PASS |
| Covariance half0_H | 0 (exact) | PASS |
| Covariance half0_B | 9.00e-12 | PASS |
| Covariance half1_H | 2.57e-10 | PASS |
| Covariance half1_B | 3.53e-11 | PASS |

**Timing:**

| Stage | 1 Node | 4 Nodes | Notes |
|-------|--------|---------|-------|
| Mean | 37.9s | 147.9s | Rank 0 post-processing dominates |
| Covariance H/B | 590.4s | 502.9s | 15% speedup with freq-splitting |

### Known Issues and Things to Investigate

1. **Mean stage slower at multi-node:** The image accumulation (~5s) is dwarfed by rank 0's post-processing (prior computation, regularization, uninvert check ~140s). The post-processing is serial and dominates total time. Need to profile what makes the post-processing so slow — `compute_relion_prior` and `post_process_from_filter` are the suspects.

2. **Covariance speedup modest (15% at 4 nodes):** The 128-box dataset is small. The 256-box dataset (16.7M volume_size vs 2M) should show more benefit since H/B computation is O(volume_size * n_freq * n_images). Need to test with `data-256-300000`.

3. **Only mean and covariance are actually distributed:** Noise refinement, PCA/SVD, and embedding run on rank 0 only. The plan identifies these as future parallelization targets but they're secondary to covariance (the bottleneck).

4. **NFS barrier reliability:** The file-based barrier uses `os.stat()` polling + `os.fsync()` on directory. Worked in testing but NFS attribute caching can cause delays. Recommend `actimeo=0` mount option on the checkpoint directory for production use.

5. **Docker container build/install race:** Multiple SLURM jobs sharing the same `.pixi` env on shared filesystem can conflict. Current fix: rank 0 installs first (with stale editable artifact cleanup), other ranks wait via marker file. Works but fragile — consider pre-building pixi env in Docker image.

6. **Pixi editable install conflicts:** Concurrent jobs produce different `__editable__*recovar*` files that go stale. Current mitigation: `rm -f __editable__*recovar*` before install. Root fix: install recovar non-editable in Docker image, or use per-job pixi env paths.

7. **CPU bind errors on some nodes:** `srun` fails with "Unable to satisfy cpu bind request" on nodes with different CPU topologies. Fixed with `--cpu-bind=none` but may reduce performance on NUMA systems.

8. **Tilt series not supported in distributed mode:** Image splitting must respect tilt-series boundaries. Deferred to Phase 2.

9. **Checkpoint disk usage:** Covariance H/B partials for 256-box can be 10-40 GB each. With 4 nodes × 2 halves, total checkpoint I/O is ~100-200 GB. Need to verify shared filesystem bandwidth is sufficient.

10. **`write_partial` had a bug with np.save:** `np.save` appends `.npy` extension, causing atomic rename to fail. Fixed by using `.tmp.npy` suffix. Watch for similar issues with `np.lib.format.open_memmap` for large arrays (>1GB path).

### Multi-Node Architecture

```
submit_job.sh <action>
  -> sbatch with --nodes=N, --ntasks-per-node=1
    -> SLURM allocates N nodes
      -> Head node: builds Docker + saves tarball
        -> srun launches run_node_container.sh on each node
          -> Each node: loads tarball, rank 0 installs pixi env
            -> All ranks: docker run with SLURM_PROCID/NTASKS/JOB_ID
              -> recovar pipeline_distributed (or test stage runner)
                -> Stage-by-stage execution with file-based barriers
```

### Parallelization Strategy

| Stage | Strategy | Notes |
|-------|----------|-------|
| Setup | Rank 0 only | Fast, broadcasts config |
| Mean | Image-split across nodes | Each rank accumulates partial ft_y/ft_ctf, rank 0 reduces + post-processes |
| Mask | Rank 0 only | Fast, no image iteration |
| Noise + Variance | Rank 0 only | Complex internal loop, deferred |
| Covariance H/B | Frequency-split across nodes | 2 halves × N/2 freq ranges; multi-GPU within each node for image batches |
| Regularization + PCA | Rank 0 only | Operates on reduced H/B matrices |
| Projected Covariance | Rank 0 only | Small accumulators (~800MB), deferred |
| Embedding | Rank 0 only | Fast relative to covariance, deferred |
| Save | Rank 0 only | Writes params.pkl, embeddings.pkl, volumes |
