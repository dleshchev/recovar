# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

**RECOVAR** — JAX/CUDA package for cryo-EM/cryo-ET conformational heterogeneity (PCA on covariance, latent embedding, volume reconstruction). Build manager: Pixi (conda). Entry point: `recovar <command>` — auto-discovers modules under `recovar/commands/`. Primary command: `recovar pipeline`. GitHub: https://github.com/ma-gilles/recovar.

## Common Commands

`pixi install` / `pixi shell`. All workloads (datasets, pipeline, profiling, multi-GPU/multi-node tests) are pixi tasks defined in `pixi.toml [tasks]`; consult that file for the full list rather than mirroring it here. Frequently used: `pipeline-{small,large}[-lazy]`, `test-{1,2,4,8}gpu`, `profile-pipeline-{small,large}[-full|-no-hb|-with-io]`, `profile-{1,2,4}gpu[-256]`, `create-dataset-{small,large}`.

## Job Submission

**You cannot run GPU workloads directly — submit via `./submit_job.sh <action>` (sbatch wrapper).** Submission chain:

```
./submit_job.sh <action>
  -> sbatch (--gpus-per-node=N --nodes=N) — generates script in scripts/job_scripts/
    -> on x86_64 GPU node: build/check Docker (recovar:latest)
      -> docker run --runtime=nvidia (auto-discovers SLURM GPUs; no --gpus flag)
        -> pixi install && pixi run <task>
```

Run `./submit_job.sh` with no args for the action list. Common: `pipeline-{small,large}`, `test-2gpu`, `profile-2gpu`, `create-{small,large}`, `smoke-gpu`. Multi-node: see "Distributed" below.

ARM exclusion: pixi env is `linux-64` only; ARM A100 nodes are excluded via `SLURM_EXCLUDE` in `submit_job.sh`. If a job dies with `unsupported-platform linux-aarch64`, add the node there.

Monitoring: `squeue -u $USER`; `tail -f scripts/output/slurm-<JOB_ID>.out`; `./submit_job.sh organize-outputs` to tidy stray logs.

## Architecture

Pipeline flow: input (STAR/MRCS/CS) → dataset loading (lazy supported) → preprocessing (CTF/mask/norm) → covariance estimation → PCA/SVD → embedding → heterogeneous volume reconstruction → output. Hottest path: `covariance_estimation.py`.

| Module | Purpose |
|---|---|
| `covariance_estimation.py` | Covariance matrix; performance-critical |
| `principal_components.py` | PCA/SVD on covariance |
| `embedding.py` | Latent space assignment |
| `homogeneous.py` | Mean volume |
| `core.py` | Low-level cryo-EM ops (CTF, Fourier slicing) |
| `noise.py` | Noise model |
| `simulator.py`, `dataset.py`, `cryo_dataset.py` | Synthetic data, dataset I/O |
| `multi_gpu_utils.py` | Multi-GPU coordination |

JIT'd JAX kernels throughout. NVTX annotations for Nsight. Batch sizes derived from GPU memory in `utils.py`. Multi-GPU is image-level parallel. Formats: RELION STAR, CryoSPARC CS, MRC/MRCS, plain text, cryoDRGN (compat layers in `starfile.py`, `cryodrgn_load.py`, `image_loader.py`).

## Distributed Pipeline (branch `multinode-dev`)

`recovar pipeline_distributed` runs the same pipeline across N ranks. Two transports: file-based (default, NFS partials + barrier files) and OpenMPI (`RECOVAR_MPI=1`, in-memory collectives). The file path is preserved for rollback; MPI is the active path.

### New modules

| Module | Purpose |
|---|---|
| `recovar/stages.py` | Pipeline decomposed into stage functions |
| `recovar/distributed.py` | Coordination: barriers, partial I/O, broadcasts (dispatches to MPI when `RECOVAR_MPI=1`) |
| `recovar/stage_checkpoint.py` | Per-stage save/load + DONE markers |
| `recovar/distributed_stages.py` | Distributed wrappers per stage (image-split mean, freq-split covariance) |
| `recovar/commands/pipeline_distributed.py` | Entry point |

`compute_regularized_covariance_columns` (in `covariance_estimation.py`) is split into `compute_both_H_B` + `regularize_covariance_columns[_in_batch]`. `principal_components.pick_covariance_frequencies` is extracted so distributed code picks frequencies independently. H/B depends on `dilated_volume_mask`, not `focus_mask` — computed once and reused across focus masks.

### Submission

```bash
# File-based path
./submit_job.sh dist-small-{1,2,4}node      # 128x100k
# MPI path (RECOVAR_MPI=1, set automatically)
./submit_job.sh dist-small-mpi-{1,2,4}node  # 128x100k, ~40-75min
./submit_job.sh dist-large-mpi-{1,2,4}node  # 256x300k lazy, --mem=500G
./submit_job.sh smoke-mpi-gpu               # MPI + JAX-GPU smoke (2 nodes, 30m)
# Stage tests
./submit_job.sh test-stage-{ref,mean-{1,2},cov-{1,2,4},compare}
./submit_job.sh profile-stage-cov-{1,2}     # nsys + NVTX
```

Dataset selection passes through wrapper env vars (`RECOVAR_IMG_SIZE`, `RECOVAR_N_IMAGES`, `RECOVAR_LAZY`); see `scripts/run_pipeline_distributed_mpi.py`.

### Parallelization

Setup: rank 0 → bcast. Mean: image-split, rank 0 reduces+post-processes. Mask: rank 0 → bcast. Noise/variance: rank 0 (deferred). Covariance+PCA: all ranks compute distributed H/B (2 halves × N/2 freq ranges via `distributed_covariance_hb`); rank 0 does regularization+SVD+rescaling per focus mask. Embedding: rank 0 (deferred). Save: rank 0.

Two NVTX domains: `compute_H_B` (JAX kernels, GPU transfers) and `distributed` (partial I/O, barriers, assembly, checkpoint).

### MPI Launcher

Cluster has PMIx 4 / OMPI 4.1.x ABI mismatch and GSSAPI auth (no usable user SSH keypair). Launcher uses sshd-in-container + mpirun-with-ssh:

```
submit_docker_mpi_job (#SBATCH --nodes=N --ntasks-per-node=1 --mem=...)
  head: docker build (always) + docker save tarball
  srun → run_node_container_mpi.sh per rank
    cp tarball to /tmp, docker load, start sshd (per-job ed25519 key, port=30000+JOBID%30000)
    rank 0: mpirun --mca plm_rsh_agent "ssh -p $PORT -i $KEY" -x KEY=VAL ...
```

Files: `scripts/run_pipeline_distributed_mpi.py`, `scripts/run_node_container_mpi.sh`, `submit_job.sh:submit_docker_mpi_job`.

### Critical fixes (do not regress)

1. **Head always docker-builds** before saving tarball. Stale `recovar:latest` predating the openssh-server install propagates via tarball and silently fails sshd. (`submit_job.sh:submit_docker_mpi_job`)
2. **Workers cp tarball to `/tmp` before `docker load`.** Direct NFS read can race with head's save → "stale file handle". (`run_node_container_mpi.sh`)
3. **`mpirun -x KEY` for every EXTRA_ENV key.** mpirun doesn't propagate process env to remote ranks via ssh; sshd strips most. Launcher emits `-x` for every EXTRA_ENV key plus PATH/LD_LIBRARY_PATH/HOME/KMP_DUPLICATE_LIB_OK. (`run_node_container_mpi.sh`)
4. **F-order H/B allocation** at `covariance_estimation.py:530-531`. Column-axis Gatherv on a C-order array silently sends wrong bytes.
5. **Contiguous-column derived datatype** for Gatherv/Send/Recv at `distributed_stages.py:660-700`. At 256-box, `volume_size × n_frequencies ≈ 5e9` overflows MPI int32 count. Solution: `mpi_dtype.Create_contiguous(volume_size).Commit()` so counts/displs are in column units.
6. **Output dir name uses `MPI.COMM_WORLD.Get_size()`** — `SLURM_JOB_NUM_NODES` not propagated through `mpirun -x`. (`run_pipeline_distributed_mpi.py`)
7. **`mark_complete()` AFTER barrier** in every distributed stage wrapper. Otherwise rank 0 can write DONE before the barrier, rank 1 sees `is_complete()==True` in the orchestrator and skips the stage (and its barrier), then desyncs.

### Memory and node selection

- `--mem=110G` default (fits 128-box). `--mem=500G` for `dist-large-mpi-*` (256-box H/B ≈ 160 GB).
- 256-box only schedules on 1TB-RAM A100-80GB-PCIe nodes; multi-node 256-box jobs need same partition.
- **Excluded** (`SLURM_EXCLUDE`): `ipp1-2160`, `ipp1-1744` (CUDA host alloc fails on low-FreeMem hosts during PCA reg); `ipp1-2029`, `ipp1-2030` (rank 0 SIGKILLed within seconds of `distributed_covariance_hb` — driver/pinned-memory issue).

### Verification (vs `pipeline-{small,large}` reference)

`scripts/compare_mpi_vs_reference.py` compares `picked_frequencies` as a set: `rel = |ref Δ cand| / max(|ref|,|cand|)`. (Element-wise comparison was misleading — picks are integer vector indices spanning ~vol_size, so a single near-tie sort flip looks like rel ≈ 0.03.)

| Run | Wallclock | cov_zs (rel) | picked_frequencies | zs |
|---|---|---|---|---|
| 1-node MPI small | 49m | 1.6-2.8e-3 | 0/300 (set-equal) | 1.5e-3 |
| 2-node MPI small | 1h14m | 1.6-3.0e-3 | 0/300 (byte-identical to 1-node) | 1.5e-3 |
| 4-node MPI small | 41m | 1.6-3.2e-3 | 2/300 (1 changed pick, rel=6.7e-3) | ≤2.1e-3 |
| Legacy 1-node | — | — | 2/300 (1 changed pick) | — |
| 1-node MPI large | 5h03m | matches at zdim 2; 0.4-0.93 at zdim 10/20 | matches | ~2.0 (sign flip) |

Small-dataset deviations are all fp32-band (JAX reduction non-associativity in `compute_variance` produces ~1 ULP noise that flips near-tie picks via `jnp.argsort`). Large-dataset eigenvectors at zdim ≥ 10 are sign-flipped/in-subspace-rotated due to repeat eigenvalues — element-wise tolerance is the wrong metric there; principal-angle would be appropriate.

### Known issues (active)

1. **NFS I/O dominates the file-based covariance path** at higher rank counts (writing/reading 2.5–5 GB partials via NFS, 20–125s each). 128-box: I/O exceeds compute savings at 4+ ranks. The MPI path bypasses partials entirely.
2. **Assembly OOM on low-RAM nodes** (32–64 GB systems can't hold assembled H/B; 20 GB for 128-box, ~160 GB for 256-box). Mitigated by `_assemble_half_to_memmap`. Use 128 GB+ nodes or `submit_singlenode_multirank_job`.
3. **OpenMP duplicate-library crash on multi-rank-per-node.** Workaround: `KMP_DUPLICATE_LIB_OK=TRUE` in `run_node_container.sh`.
4. **Mean stage slower at multi-node** — rank 0 post-processing (~140 s) dominates the ~5 s parallel accumulation.
5. **Noise and embedding still rank-0 only** (deferred).
6. **Tilt series not supported in distributed mode** (image-split must respect tilt boundaries; deferred).
7. **NFS barrier reliability.** File-based barrier polls `os.stat()`; NFS attribute caching can delay. Timeout 1800 s. The MPI path uses `MPI.Barrier` and avoids this.
8. **Final `pipeline_complete` cleanup race.** Rank 0 cleans `checkpoint_dir` after the final barrier; rank 1 may still poll for rank 0's marker and never see it. Output is correct but the job hangs until SLURM kills it. Workaround: `--keep-checkpoints` or cancel after rank 0 reports completion.
9. **Docker install race** when concurrent SLURM jobs share `.pixi`. Rank 0 installs first, others wait via marker file.
10. **2/4-node large MPI jobs blocked by SLURM partitioning.** 256-box needs same-partition pairs of 1TB+a100-80gb-pcie nodes; the relevant partition has ~3 schedulable such nodes (one excluded for SIGKILL). Wait or relax `--mem`.
11. **Variable per-node performance.** ipp1-216x are 3–4× slower at SVD than ipp1-1xxx; default time limit raised to 3h.
12. **Legacy file-based regression: sign-flipped eigenvectors at `RECOVAR_MPI=0` (`dist-small-1node`).** Phase 4a's `order='F'` H/B allocation (load-bearing for MPI Gatherv) cascades through `regularize_covariance_columns_in_batch`. Documented regression in the path being retired; not blocking the MPI path.
13. **picked_frequencies near-tie pick flip is fp32 noise, not orchestration drift.** Set diff is 0/300 for MPI 1+2-node vs reference and 2/300 for MPI 4-node and legacy 1-node. Cross-comparison of `variance_est['combined']` shows rel ∈ [7.7e-7, 4.0e-6] across all 5 paths (ref/legacy/mpi{1,2,4}-node) — JAX reduction non-associativity in `compute_variance` is the noise source. Stable-sort tiebreaks won't help; values aren't equal, just close in fp32.
