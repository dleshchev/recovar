"""
Stage checkpoint management for RECOVAR pipeline.

Each pipeline stage can save/load its results to a checkpoint directory,
enabling restart from any stage.
"""

import os
import json
import pickle
import time
import logging
import numpy as np
import nvtx
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NVTX_DOMAIN_DIST = "distributed"


class StageCheckpoint:
    """Manages checkpoint directory for a single pipeline stage."""

    def __init__(self, checkpoint_root: str, stage_name: str):
        """Create checkpoint for a stage.

        Args:
            checkpoint_root: Root checkpoint directory
            stage_name: Stage identifier (e.g., 'stage_02_mean_pass0')
        """
        self._dir = os.path.join(checkpoint_root, stage_name)
        self._stage_name = stage_name
        os.makedirs(self._dir, exist_ok=True)

    @property
    def dir(self) -> str:
        """Checkpoint directory path."""
        return self._dir

    @property
    def stage_name(self) -> str:
        """Stage identifier."""
        return self._stage_name

    def save_array(self, name: str, arr: np.ndarray) -> str:
        """Save numpy array. Uses memmap for arrays > 1 GB."""
        path = os.path.join(self._dir, f"{name}.npy")

        if arr.nbytes > 1_000_000_000:
            logger.info(f"Saving large array via memmap: {name} "
                       f"(shape={arr.shape}, dtype={arr.dtype}, "
                       f"size={arr.nbytes / 1e9:.1f} GB)")
            fp = np.lib.format.open_memmap(path, mode='w+', dtype=arr.dtype, shape=arr.shape)
            fp[:] = arr
            del fp
        else:
            logger.info(f"Saving array: {name} (shape={arr.shape}, dtype={arr.dtype})")
            np.save(path, arr)

        return path

    def load_array(self, name: str) -> np.ndarray:
        """Load numpy array from checkpoint."""
        path = os.path.join(self._dir, f"{name}.npy")
        logger.info(f"Loading array: {name}")
        return np.load(path)

    def save_config(self, config: dict) -> str:
        """Save config dict as JSON."""
        path = os.path.join(self._dir, "config.json")
        logger.info(f"Saving config: {list(config.keys())}")
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        return path

    def load_config(self) -> dict:
        """Load config from JSON."""
        path = os.path.join(self._dir, "config.json")
        with open(path, 'r') as f:
            return json.load(f)

    def save_object(self, name: str, obj: Any) -> str:
        """Save arbitrary Python object as pickle."""
        path = os.path.join(self._dir, f"{name}.pkl")
        st = time.time()
        with nvtx.annotate(f"save_object_{name}", color="purple",
                           domain=NVTX_DOMAIN_DIST):
            logger.info(f"Saving object: {name} (type={type(obj).__name__})")
            with open(path, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / 1e6
        logger.info(f"Saved object: {name} ({size_mb:.0f} MB, {time.time()-st:.1f}s)")
        return path

    def load_object(self, name: str) -> Any:
        """Load object from pickle."""
        path = os.path.join(self._dir, f"{name}.pkl")
        st = time.time()
        with nvtx.annotate(f"load_object_{name}", color="blue",
                           domain=NVTX_DOMAIN_DIST):
            logger.info(f"Loading object: {name}")
            with open(path, 'rb') as f:
                result = pickle.load(f)
        size_mb = os.path.getsize(path) / 1e6
        logger.info(f"Loaded object: {name} ({size_mb:.0f} MB, {time.time()-st:.1f}s)")
        return result

    def is_complete(self) -> bool:
        """Check if stage has completed (DONE marker exists)."""
        return os.path.exists(os.path.join(self._dir, "DONE"))

    def mark_complete(self) -> None:
        """Write DONE marker to indicate stage completion."""
        path = os.path.join(self._dir, "DONE")
        with open(path, 'w') as f:
            f.write("")
        logger.info(f"Stage '{self._stage_name}' marked complete")

    def clear(self) -> None:
        """Remove all files in checkpoint directory (for restart).
        Keeps the directory itself."""
        logger.warning(f"Clearing checkpoint: {self._stage_name}")
        for fname in os.listdir(self._dir):
            fpath = os.path.join(self._dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)

    def partial_path(self, rank: int, name: str) -> str:
        """Return path for a partial result file.

        Args:
            rank: Node rank (zero-padded to 4 digits for sort order)
            name: Partial result name

        Returns:
            Path like {dir}/partial_{rank:04d}_{name}.npy
        """
        return os.path.join(self._dir, f"partial_{rank:04d}_{name}.npy")

    def list_files(self) -> List[str]:
        """List all files in checkpoint directory."""
        if not os.path.exists(self._dir):
            return []
        return sorted(os.listdir(self._dir))

    def has_array(self, name: str) -> bool:
        """Check if a named array exists in checkpoint."""
        return os.path.exists(os.path.join(self._dir, f"{name}.npy"))

    def has_object(self, name: str) -> bool:
        """Check if a named object exists in checkpoint."""
        return os.path.exists(os.path.join(self._dir, f"{name}.pkl"))

    def __repr__(self) -> str:
        status = "complete" if self.is_complete() else "incomplete"
        n_files = len(self.list_files())
        return f"StageCheckpoint('{self._stage_name}', {status}, {n_files} files)"
