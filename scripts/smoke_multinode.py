#!/usr/bin/env python3
"""Multi-node smoke test: verify SLURM env vars and GPU access."""
import os
rank = os.environ.get("SLURM_PROCID", "?")
ntasks = os.environ.get("SLURM_NTASKS", "?")
job_id = os.environ.get("SLURM_JOB_ID", "?")
node = os.uname().nodename
print(f"Rank {rank}/{ntasks} on {node} (job {job_id})")

import jax
devices = jax.devices("gpu")
print(f"Rank {rank}: {len(devices)} GPU(s): {[str(d) for d in devices]}")
print(f"Rank {rank}: SMOKE TEST PASSED")
