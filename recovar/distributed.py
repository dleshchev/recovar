"""
Multi-node distributed coordination primitives for RECOVAR.

File-based coordination via shared filesystem. No JAX dependency.
For NFS mounts, recommend actimeo=0 on the checkpoint directory.
"""

import os
import glob
import json
import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    """Identity of this node in a distributed run."""
    rank: int
    world_size: int
    job_id: str


@dataclass
class Assignment:
    """Work assignment for image-parallel stages."""
    half: int  # 0 or 1
    image_indices: np.ndarray


@dataclass
class FrequencyAssignment:
    """Work assignment for frequency-parallel stages."""
    half: int  # 0 or 1
    freq_start: int
    freq_end: int


def get_node_config_from_env() -> NodeConfig:
    """Read node identity from SLURM environment variables.
    Falls back to single-node defaults when not in SLURM."""
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    config = NodeConfig(rank=rank, world_size=world_size, job_id=job_id)
    logger.info(f"Node config: rank={config.rank}, world_size={config.world_size}, job_id={config.job_id}")
    return config


def _split_evenly(n_items: int, n_parts: int) -> List[Tuple[int, int]]:
    """Split n_items into n_parts contiguous ranges. Returns list of (start, end) tuples."""
    chunk = n_items // n_parts
    remainder = n_items % n_parts
    ranges = []
    start = 0
    for i in range(n_parts):
        end = start + chunk + (1 if i < remainder else 0)
        ranges.append((start, end))
        start = end
    return ranges


def compute_image_assignments(world_size: int, n_images_per_half: List[int]) -> List[Assignment]:
    """Assign image subsets to each rank for image-parallel stages.

    Args:
        world_size: Number of nodes
        n_images_per_half: [n_images_half0, n_images_half1]

    Returns:
        List of Assignment, one per rank. For world_size=1, returns single
        assignment for half 0 (caller handles both halves sequentially).
    """
    if world_size == 1:
        return [Assignment(half=0, image_indices=np.arange(n_images_per_half[0]))]

    if world_size == 2:
        return [
            Assignment(half=0, image_indices=np.arange(n_images_per_half[0])),
            Assignment(half=1, image_indices=np.arange(n_images_per_half[1])),
        ]

    # world_size >= 3: split evenly between halves
    nodes_per_half = world_size // 2
    # If odd world_size, half 0 gets the extra node
    nodes_half0 = nodes_per_half + (world_size % 2)
    nodes_half1 = nodes_per_half

    assignments = []

    # Half 0: ranks 0 .. nodes_half0-1
    ranges_h0 = _split_evenly(n_images_per_half[0], nodes_half0)
    for start, end in ranges_h0:
        assignments.append(Assignment(half=0, image_indices=np.arange(start, end)))

    # Half 1: ranks nodes_half0 .. world_size-1
    ranges_h1 = _split_evenly(n_images_per_half[1], nodes_half1)
    for start, end in ranges_h1:
        assignments.append(Assignment(half=1, image_indices=np.arange(start, end)))

    for rank, a in enumerate(assignments):
        logger.debug(f"Rank {rank}: half={a.half}, n_images={len(a.image_indices)}")

    return assignments


def compute_frequency_assignments(world_size: int, n_frequencies: int) -> List[FrequencyAssignment]:
    """Assign frequency ranges to each rank for frequency-parallel stages.

    Args:
        world_size: Number of nodes
        n_frequencies: Total number of picked frequency columns

    Returns:
        List of FrequencyAssignment, one per rank.
    """
    if world_size == 1:
        return [FrequencyAssignment(half=0, freq_start=0, freq_end=n_frequencies)]

    if world_size == 2:
        return [
            FrequencyAssignment(half=0, freq_start=0, freq_end=n_frequencies),
            FrequencyAssignment(half=1, freq_start=0, freq_end=n_frequencies),
        ]

    # world_size >= 3: split frequencies within each half
    nodes_per_half = world_size // 2
    nodes_half0 = nodes_per_half + (world_size % 2)
    nodes_half1 = nodes_per_half

    assignments = []

    ranges_h0 = _split_evenly(n_frequencies, nodes_half0)
    for start, end in ranges_h0:
        assignments.append(FrequencyAssignment(half=0, freq_start=start, freq_end=end))

    ranges_h1 = _split_evenly(n_frequencies, nodes_half1)
    for start, end in ranges_h1:
        assignments.append(FrequencyAssignment(half=1, freq_start=start, freq_end=end))

    for rank, fa in enumerate(assignments):
        logger.debug(f"Rank {rank}: half={fa.half}, freqs=[{fa.freq_start}:{fa.freq_end}]")

    return assignments


def write_partial(path: str, array: np.ndarray) -> None:
    """Atomically write array to .npy file (write .tmp, then rename).
    Uses memmap for arrays > 1 GB to avoid memory doubling."""
    # Ensure path ends with .npy (np.save appends it if missing)
    if not path.endswith('.npy'):
        path = path + '.npy'
    # Use .tmp.npy so np.save doesn't add another .npy extension
    tmp_path = path[:-4] + '.tmp.npy'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if array.nbytes > 1_000_000_000:
        logger.info(f"Writing large partial ({array.nbytes / 1e9:.1f} GB) via memmap: {path}")
        fp = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=array.dtype, shape=array.shape)
        fp[:] = array
        del fp  # flush
    else:
        np.save(tmp_path, array)

    os.rename(tmp_path, path)
    logger.debug(f"Wrote partial: {path} (shape={array.shape}, dtype={array.dtype})")


def read_and_reduce_partials(directory: str, pattern: str, reduce: str = 'sum') -> np.ndarray:
    """Load partial result files matching pattern, reduce by summation.

    Args:
        directory: Directory containing partial files
        pattern: Glob pattern (e.g., '*_half0_ft_y.npy')
        reduce: Reduction operation ('sum' only for now)

    Returns:
        Reduced array
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")

    logger.info(f"Reducing {len(files)} partials matching '{pattern}'")

    result = np.load(files[0])
    for f in files[1:]:
        result = result + np.load(f)

    return result


def read_and_concat_partials(directory: str, pattern: str, axis: int = 0) -> np.ndarray:
    """Load partial result files matching pattern, concatenate along axis.
    Files are sorted by name (zero-padded rank ensures correct order).

    Args:
        directory: Directory containing partial files
        pattern: Glob pattern (e.g., '*_half0_H.npy')
        axis: Concatenation axis

    Returns:
        Concatenated array
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")

    logger.info(f"Concatenating {len(files)} partials matching '{pattern}' along axis {axis}")

    arrays = [np.load(f) for f in files]
    return np.concatenate(arrays, axis=axis)


def _fsync_directory(directory: str) -> None:
    """Fsync a directory to flush metadata to NFS server."""
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some filesystems don't support fsync on directories
        pass


def barrier(directory: str, name: str, world_size: int, job_id: str,
            rank: int, timeout: float = 600) -> None:
    """File-based barrier synchronization.

    Each rank writes a marker file, then polls until all ranks have written theirs.
    Uses os.stat() per file (more reliable on NFS than os.listdir()).

    Args:
        directory: Shared directory for barrier files
        name: Barrier name (unique per synchronization point)
        world_size: Total number of ranks
        job_id: SLURM job ID (prevents stale files from previous runs)
        rank: This rank's ID
        timeout: Seconds before raising TimeoutError
    """
    if world_size == 1:
        return  # No synchronization needed

    os.makedirs(directory, exist_ok=True)

    # Write this rank's barrier file
    barrier_file = os.path.join(directory, f"barrier_{job_id}_{name}_{rank:04d}")
    with open(barrier_file, 'w') as f:
        f.write("")

    # Fsync directory to push to NFS server
    _fsync_directory(directory)

    # Poll for all ranks
    start_time = time.time()
    expected_files = [
        os.path.join(directory, f"barrier_{job_id}_{name}_{r:04d}")
        for r in range(world_size)
    ]

    while True:
        all_present = True
        for ef in expected_files:
            try:
                os.stat(ef)
            except FileNotFoundError:
                all_present = False
                break

        if all_present:
            logger.debug(f"Barrier '{name}' passed (rank {rank})")
            return

        elapsed = time.time() - start_time
        if elapsed > timeout:
            # Report which ranks are missing
            missing = []
            for r, ef in enumerate(expected_files):
                try:
                    os.stat(ef)
                except FileNotFoundError:
                    missing.append(r)
            raise TimeoutError(
                f"Barrier '{name}' timed out after {timeout}s. "
                f"Missing ranks: {missing}. "
                f"This rank: {rank}, world_size: {world_size}, job_id: {job_id}"
            )

        time.sleep(0.5)


def cleanup_barriers(directory: str, name: str, job_id: str) -> None:
    """Remove barrier files for a given barrier name and job."""
    pattern = os.path.join(directory, f"barrier_{job_id}_{name}_*")
    for f in glob.glob(pattern):
        try:
            os.remove(f)
        except OSError:
            pass


def is_stage_complete(stage_dir: str) -> bool:
    """Check if a stage has completed (DONE marker exists)."""
    return os.path.exists(os.path.join(stage_dir, "DONE"))


def mark_stage_complete(stage_dir: str) -> None:
    """Write DONE marker to indicate stage completion."""
    done_path = os.path.join(stage_dir, "DONE")
    with open(done_path, 'w') as f:
        f.write("")
    _fsync_directory(stage_dir)
    logger.info(f"Stage complete: {stage_dir}")


def broadcast_value(directory: str, name: str, value, rank: int,
                    world_size: int, job_id: str) -> Union[np.ndarray, dict]:
    """Rank 0 writes a value, all ranks read it after barrier.

    Args:
        directory: Shared directory
        name: Value name (used for filename)
        value: Value to broadcast (np.ndarray or dict/scalar)
        rank: This rank's ID
        world_size: Total number of ranks
        job_id: SLURM job ID

    Returns:
        The broadcast value (loaded from disk on all ranks)
    """
    os.makedirs(directory, exist_ok=True)

    if rank == 0:
        if isinstance(value, np.ndarray):
            path = os.path.join(directory, f"{name}.npy")
            write_partial(path, value)
        else:
            path = os.path.join(directory, f"{name}.json")
            with open(path, 'w') as f:
                json.dump(value, f)

    barrier(directory, f"broadcast_{name}", world_size, job_id, rank)

    # All ranks load
    npy_path = os.path.join(directory, f"{name}.npy")
    json_path = os.path.join(directory, f"{name}.json")

    if os.path.exists(npy_path):
        return np.load(npy_path)
    elif os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Broadcast value '{name}' not found in {directory}")
