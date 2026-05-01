"""Compare a distributed MPI pipeline output against the single-node reference.

Pass criterion (per the migration plan): all numerical outputs match within
1e-3 relative tolerance. Reports a per-item diff and a single PASS/FAIL line
at the end.

Usage (run inside the pixi env):
    python scripts/compare_mpi_vs_reference.py \\
        --reference data-128-100000/test_dataset/pipeline_output \\
        --candidate data-128-100000/test_dataset/pipeline_distributed_mpi_output
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np


RTOL = 1e-3


def _is_array_like(x) -> bool:
    return isinstance(x, np.ndarray)


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Relative max-abs error: ||a-b||_inf / max(||a||_inf, ||b||_inf, eps).
    Handles complex arrays via abs."""
    if a.shape != b.shape:
        return float("inf")
    a64 = a.astype(np.float64) if a.dtype != np.complex128 else a
    b64 = b.astype(np.float64) if b.dtype != np.complex128 else b
    diff = np.abs(a64 - b64).max()
    scale = max(np.abs(a64).max(), np.abs(b64).max(), 1e-30)
    return float(diff / scale)


def _walk_compare(label: str, ref, cand, results: list):
    """Walk pickled objects in parallel, recording per-leaf relative diffs."""
    if isinstance(ref, dict) and isinstance(cand, dict):
        # Sort keys by their string representation so dicts with mixed-type
        # keys (e.g., {"foo": ..., 4: ...}) don't blow up.
        keys = sorted(set(ref) | set(cand), key=lambda x: (type(x).__name__, str(x)))
        for k in keys:
            sub = f"{label}.{k}" if label else str(k)
            if k not in ref:
                results.append((sub, "MISSING_REFERENCE", None))
            elif k not in cand:
                results.append((sub, "MISSING_CANDIDATE", None))
            else:
                _walk_compare(sub, ref[k], cand[k], results)
        return
    if isinstance(ref, (list, tuple)) and isinstance(cand, (list, tuple)):
        if len(ref) != len(cand):
            results.append((label, f"LEN_MISMATCH ref={len(ref)} cand={len(cand)}", None))
            return
        for i, (a, b) in enumerate(zip(ref, cand)):
            _walk_compare(f"{label}[{i}]", a, b, results)
        return
    if _is_array_like(ref) and _is_array_like(cand):
        if ref.shape != cand.shape:
            results.append((label, f"SHAPE_MISMATCH ref={ref.shape} cand={cand.shape}", None))
            return
        if ref.size == 0:
            results.append((label, "EMPTY", 0.0))
            return
        if ref.dtype.kind == "O":
            # 0-d object array (e.g. argparse.Namespace wrapped in np.array)
            results.append((label, "OBJECT_ARRAY",
                            0.0 if str(ref.tolist()) == str(cand.tolist()) else None))
            return
        # picked_frequencies is an array of integer vector indices. Comparing
        # element-wise with max-abs over the integer values amplifies a single
        # near-tie sort flip into a huge "rel" (e.g. swapping two adjacent picks
        # whose vec indices differ by ~70k looks like rel ≈ 0.03 against a
        # vol_size-scale max). The selection is set-valued, so compare it that
        # way: |ref Δ cand| / max(|ref|, |cand|).
        if label.endswith("picked_frequencies"):
            sa = set(map(int, ref.ravel()))
            sb = set(map(int, cand.ravel()))
            sym = len(sa ^ sb)
            denom = max(len(sa), len(sb), 1)
            results.append((label, f"set_sym_diff={sym}", float(sym) / denom))
            return
        results.append((label, ref.dtype.name, _rel_err(ref, cand)))
        return
    if isinstance(ref, (int, float, np.number)) and isinstance(cand, (int, float, np.number)):
        scale = max(abs(ref), abs(cand), 1e-30)
        rel = abs(float(ref) - float(cand)) / scale if scale else 0.0
        results.append((label, type(ref).__name__, rel))
        return
    # Last-resort: compare via repr (handles argparse.Namespace etc).
    try:
        if ref == cand:
            results.append((label, "EQUAL", 0.0))
            return
    except Exception:
        pass
    if repr(ref) == repr(cand):
        results.append((label, "EQUAL_REPR", 0.0))
    else:
        results.append((label, f"NEQ kind={type(ref).__name__}", None))


def _load_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, type=Path,
                   help="Single-node pipeline_output directory")
    p.add_argument("--candidate", required=True, type=Path,
                   help="MPI pipeline_distributed_mpi_output directory")
    p.add_argument("--rtol", type=float, default=RTOL,
                   help=f"Relative tolerance (default {RTOL})")
    args = p.parse_args()

    ref_model = args.reference / "model"
    cand_model = args.candidate / "model"

    if not ref_model.exists():
        print(f"ERROR: reference model dir missing: {ref_model}", file=sys.stderr)
        return 2
    if not cand_model.exists():
        print(f"ERROR: candidate model dir missing: {cand_model}", file=sys.stderr)
        return 2

    pickle_files = sorted(
        f.name for f in ref_model.glob("*.pkl")
        if (cand_model / f.name).exists()
    )

    print(f"Comparing {len(pickle_files)} pickle files (rtol={args.rtol}):")
    print(f"  reference: {ref_model}")
    print(f"  candidate: {cand_model}")

    all_results: list[tuple[str, str, float | None]] = []
    for fname in pickle_files:
        print(f"\n=== {fname} ===")
        ref_obj = _load_pickle(ref_model / fname)
        cand_obj = _load_pickle(cand_model / fname)
        per_file: list[tuple[str, str, float | None]] = []
        _walk_compare(fname, ref_obj, cand_obj, per_file)
        worst = 0.0
        worst_label = None
        for label, kind, rel in per_file:
            tag = "OK" if (rel is not None and rel <= args.rtol) else "FAIL"
            print(f"  [{tag:4}] {label}: dtype={kind} rel={rel}")
            if rel is not None and rel > worst:
                worst = rel
                worst_label = label
        all_results.extend(per_file)
        if worst_label is not None:
            print(f"  --> file worst rel: {worst:.3e} at {worst_label}")

    fails = [(l, k, r) for l, k, r in all_results
             if r is None or r > args.rtol]
    print("\n" + "=" * 60)
    print(f"Total leaves compared: {len(all_results)}")
    print(f"Total failures: {len(fails)}")
    if fails:
        print("\nFAILURES:")
        for label, kind, rel in fails[:30]:
            print(f"  {label}: dtype={kind} rel={rel}")
        if len(fails) > 30:
            print(f"  ... ({len(fails) - 30} more)")
        print("\nFAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
