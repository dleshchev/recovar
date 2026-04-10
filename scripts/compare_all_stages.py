#!/usr/bin/env python3
"""Compare all available stage test outputs and print a summary."""
import pickle, numpy as np, json, os

base = os.environ.get("DATASET_DIR", "/workspace/data-128-100000/test_dataset")

def compare_arrays(name, a, b, tol=1e-3):
    a, b = np.array(a), np.array(b)
    mx = max(np.max(np.abs(a)), 1e-10)
    rel = np.max(np.abs(a - b)) / mx
    status = "PASS" if rel < tol else "FAIL"
    print(f"  {name}: rel_err={rel:.2e} [{status}]")
    return rel < tol

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

all_pass = True
print("=" * 60)
print("MEAN COMPARISON")
print("=" * 60)
for na, nb in [(1, 2), (1, 4), (2, 4)]:
    da = f"{base}/stage_test_mean_{na}node/stage_02_mean"
    db = f"{base}/stage_test_mean_{nb}node/stage_02_mean"
    if not os.path.exists(f"{da}/means.pkl") or not os.path.exists(f"{db}/means.pkl"):
        print(f"  {na}v{nb}: SKIP (missing)")
        continue
    print(f"  --- {na} node vs {nb} nodes ---")
    ma, mb = load_pkl(f"{da}/means.pkl"), load_pkl(f"{db}/means.pkl")
    for key in ['combined', 'corrected0', 'corrected1']:
        if key in ma and key in mb:
            ok = compare_arrays(f"means['{key}']", ma[key], mb[key])
            all_pass = all_pass and ok
    pa, pb = np.load(f"{da}/mean_prior.npy"), np.load(f"{db}/mean_prior.npy")
    ok = compare_arrays("mean_prior", pa, pb)
    all_pass = all_pass and ok

print()
print("=" * 60)
print("COVARIANCE H/B COMPARISON")
print("=" * 60)
for na, nb in [(1, 2), (1, 4), (2, 4)]:
    da = f"{base}/stage_test_covariance_hb_{na}node/stage_05_covariance"
    db = f"{base}/stage_test_covariance_hb_{nb}node/stage_05_covariance"
    if not os.path.exists(f"{da}/covariance_hb_result.pkl") or not os.path.exists(f"{db}/covariance_hb_result.pkl"):
        print(f"  {na}v{nb}: SKIP (missing)")
        continue
    print(f"  --- {na} node vs {nb} nodes ---")
    Ha, Ba = load_pkl(f"{da}/covariance_hb_result.pkl")
    Hb, Bb = load_pkl(f"{db}/covariance_hb_result.pkl")
    for h in range(2):
        ok = compare_arrays(f"half{h}_H (shape={np.array(Ha[h]).shape})", Ha[h], Hb[h])
        all_pass = all_pass and ok
        ok = compare_arrays(f"half{h}_B (shape={np.array(Ba[h]).shape})", Ba[h], Bb[h])
        all_pass = all_pass and ok

print()
print("=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
for stage_name, ckpt_subdir in [("mean", "stage_02_mean"), ("covariance", "stage_05_covariance")]:
    print(f"  {stage_name}:")
    for nodes in [1, 2, 4]:
        cfg = f"{base}/stage_test_{stage_name if stage_name != 'covariance' else 'covariance_hb'}_{nodes}node/{ckpt_subdir}/config.json"
        if os.path.exists(cfg):
            with open(cfg) as f:
                d = json.load(f)
            print(f"    {nodes} node(s): {d.get('elapsed_s', '?'):.1f}s")
        else:
            print(f"    {nodes} node(s): no data")

print()
print("=" * 60)
if all_pass:
    print("ALL COMPARISONS PASSED")
else:
    print("SOME COMPARISONS FAILED")
print("=" * 60)
