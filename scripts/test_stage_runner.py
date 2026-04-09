#!/usr/bin/env python3
"""
Test individual distributed pipeline stages with different node counts.

Usage:
  # Phase 1: Generate reference checkpoints (run once, 1 node)
  python test_stage_runner.py reference --dataset-dir /path/to/data-128-100000/test_dataset

  # Phase 2: Run a specific stage with distributed wrappers
  python test_stage_runner.py run-stage --stage mean \
      --ref-checkpoint /path/to/reference_checkpoint \
      --output-dir /path/to/test_output

  # Phase 3: Compare outputs from different node counts
  python test_stage_runner.py compare \
      --ref-dir /path/to/test_output_1node \
      --test-dir /path/to/test_output_2node \
      --stage mean

This script is meant to be called from within a SLURM job / Docker container
where SLURM_PROCID and SLURM_NTASKS are set.
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
import pickle

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("test_stage_runner")


def load_reference_state(ref_checkpoint_dir, args_override=None):
    """Load pipeline state from reference checkpoints."""
    from recovar.stage_checkpoint import StageCheckpoint
    from recovar import dataset, noise

    # Load setup
    c_setup = StageCheckpoint(ref_checkpoint_dir, "stage_00_setup")
    if not c_setup.is_complete():
        raise RuntimeError(f"Reference setup checkpoint not found at {c_setup.dir}")
    setup = c_setup.load_object("setup_result")

    # Reconstruct cryos
    lazy = args_override.lazy if args_override and hasattr(args_override, 'lazy') else False
    cryos = dataset.get_split_datasets_from_dict(
        setup['dataset_loader_dict'], setup['ind_split'], lazy)

    # Initialize noise model
    noise_model = setup['noise_model']
    for cryo in cryos:
        if noise_model == "radial":
            cryo.set_radial_noise_model(None)
        elif noise_model in ('radial_per_tilt', 'radial-per-tilt'):
            cryo.set_variable_radial_noise_model(None)

    return cryos, setup


def run_reference(args):
    """Phase 1: Run standard pipeline with checkpoints to generate reference."""
    logger.info(f"Generating reference checkpoints from dataset: {args.dataset_dir}")

    # Build a mock args namespace matching what the pipeline expects
    dataset_dir = args.dataset_dir
    image_size = args.image_size
    outdir = os.path.join(dataset_dir, "stage_test_reference")
    checkpoint_dir = os.path.join(dataset_dir, "stage_test_reference_checkpoint")

    particles = os.path.join(dataset_dir, f"particles.{image_size}.mrcs")
    ctf = os.path.join(dataset_dir, "ctf.pkl")
    poses = os.path.join(dataset_dir, "poses.pkl")

    cmd = (
        f"recovar pipeline {particles} --ctf {ctf} --poses {poses} "
        f"--mask=from_halfmaps -o {outdir} "
        f"--checkpoint-dir {checkpoint_dir} --keep-checkpoints"
    )
    logger.info(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        logger.error(f"Reference pipeline failed with exit code {ret}")
        sys.exit(1)

    logger.info(f"Reference checkpoints at: {checkpoint_dir}")
    logger.info(f"Reference output at: {outdir}")


def run_stage(args):
    """Phase 2: Run a single stage using distributed wrappers."""
    import recovar.config  # noqa: F401 — initializes JAX
    from recovar.stage_checkpoint import StageCheckpoint
    from recovar.distributed import get_node_config_from_env, barrier
    from recovar import distributed_stages, noise, stages

    node_config = get_node_config_from_env()
    stage_name = args.stage
    ref_dir = args.ref_checkpoint
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        f"Rank {node_config.rank}/{node_config.world_size}: "
        f"running stage '{stage_name}' from ref={ref_dir} to output={output_dir}"
    )

    # Load reference state
    cryos, setup = load_reference_state(ref_dir)
    batch_size = setup['batch_size']
    gpu_memory = setup['gpu_memory']
    noise_var_from_hf = setup['noise_var_from_hf']
    noise_model = setup['noise_model']
    valid_idx = setup['valid_idx']
    options = setup['options']

    # Build a minimal args namespace for stages that need it
    import argparse as ap
    pipeline_args = ap.Namespace(
        mean_fn='triangular', uninvert_data='automatic',
        mask='from_halfmaps', focus_mask=None, use_complement_mask=False,
        mask_dilate_iter=0, keep_input_mask=False,
        dilated_mask_dilation_iters=None, only_mean=False,
        new_noise_est=False, premultiplied_ctf=False,
        low_memory_option=False, very_low_memory_option=False,
        dont_use_image_mask=False, test_covar_options=False,
        keep_intermediate=False, use_reg_mean_in_contrast=True,
        outdir=output_dir, lazy=False,
    )

    st_time = time.time()

    if stage_name == "mean":
        ckpt = StageCheckpoint(output_dir, "stage_02_mean")
        means, mean_prior, uninvert = distributed_stages.distributed_mean(
            cryos, batch_size, noise_var_from_hf, pipeline_args,
            node_config, ckpt)
        elapsed = time.time() - st_time

        if node_config.rank == 0:
            # Save timing
            ckpt.save_config({
                "stage": "mean", "elapsed_s": elapsed,
                "world_size": node_config.world_size,
                "uninvert_applied": uninvert,
            })
            logger.info(f"Stage 'mean' completed in {elapsed:.1f}s (world_size={node_config.world_size})")

    elif stage_name == "noise_variance":
        # Load mean from reference
        c_mean = StageCheckpoint(ref_dir, "stage_02_mean")
        means = c_mean.load_object("means")
        uninvert = c_mean.load_config().get("uninvert_applied", False)
        if uninvert:
            for cryo in cryos:
                cryo.image_stack.mult = -1 * cryo.image_stack.mult

        # Load mask from reference
        c_mask = StageCheckpoint(ref_dir, "stage_03_mask")
        volume_mask, dilated_volume_mask, focus_masks = c_mask.load_object("mask_result")

        ckpt = StageCheckpoint(output_dir, "stage_04_noise")
        result = distributed_stages.distributed_noise_refine_and_variance(
            cryos[0], cryos, means, batch_size, dilated_volume_mask,
            pipeline_args, noise_model, node_config, ckpt)
        elapsed = time.time() - st_time

        if node_config.rank == 0:
            ckpt.save_config({
                "stage": "noise_variance", "elapsed_s": elapsed,
                "world_size": node_config.world_size,
            })
            logger.info(f"Stage 'noise_variance' completed in {elapsed:.1f}s")

    elif stage_name == "covariance_hb":
        # Load prerequisites from reference
        c_mean = StageCheckpoint(ref_dir, "stage_02_mean")
        means = c_mean.load_object("means")
        uninvert = c_mean.load_config().get("uninvert_applied", False)
        if uninvert:
            for cryo in cryos:
                cryo.image_stack.mult = -1 * cryo.image_stack.mult

        c_mask = StageCheckpoint(ref_dir, "stage_03_mask")
        volume_mask, dilated_volume_mask, focus_masks = c_mask.load_object("mask_result")

        c_noise = StageCheckpoint(ref_dir, "stage_04_noise")
        noise_result = c_noise.load_object("noise_result")
        noise_var_used = noise_result[0]
        variance_est = noise_result[1]
        noise.update_noise_variance(noise_var_used, cryos)

        # Get covariance options and picked frequencies
        from recovar import covariance_estimation
        covariance_options = covariance_estimation.get_default_covariance_computation_options(cryos[0].grid_size)

        # We need picked_frequencies. Get them from the PCA reference checkpoint
        c_pca = StageCheckpoint(ref_dir, "stage_06_pca")
        if c_pca.is_complete():
            pca_result = c_pca.load_object("pca_result")
            picked_frequencies = pca_result[3]  # (u, s, cov_cols, picked_frequencies, ...)
            covariance_options = pca_result[5]
        else:
            # Compute column selection (rank 0 only, fast)
            from recovar import principal_components
            focus_mask = focus_masks[-1]
            picked_frequencies = principal_components._select_frequencies(
                variance_est['combined'], covariance_options, cryos[0].volume_shape)

        ckpt = StageCheckpoint(output_dir, "stage_05_covariance")
        Hs, Bs = distributed_stages.distributed_covariance_hb(
            cryos, means, dilated_volume_mask, picked_frequencies,
            gpu_memory, covariance_options, node_config, ckpt)
        elapsed = time.time() - st_time

        if node_config.rank == 0:
            ckpt.save_config({
                "stage": "covariance_hb", "elapsed_s": elapsed,
                "world_size": node_config.world_size,
                "n_frequencies": len(picked_frequencies),
                "H_shapes": [h.shape for h in Hs] if Hs else [],
            })
            logger.info(f"Stage 'covariance_hb' completed in {elapsed:.1f}s")

    elif stage_name == "embedding":
        # Load all prerequisites from reference
        c_mean = StageCheckpoint(ref_dir, "stage_02_mean")
        means = c_mean.load_object("means")
        uninvert = c_mean.load_config().get("uninvert_applied", False)
        if uninvert:
            for cryo in cryos:
                cryo.image_stack.mult = -1 * cryo.image_stack.mult

        c_mask = StageCheckpoint(ref_dir, "stage_03_mask")
        volume_mask, dilated_volume_mask, focus_masks = c_mask.load_object("mask_result")

        c_noise = StageCheckpoint(ref_dir, "stage_04_noise")
        noise_result = c_noise.load_object("noise_result")
        noise_var_used = noise_result[0]
        noise.update_noise_variance(noise_var_used, cryos)

        c_pca = StageCheckpoint(ref_dir, "stage_06_pca")
        u, s, covariance_cols, picked_frequencies, column_fscs, covariance_options = \
            c_pca.load_object("pca_result")

        ckpt = StageCheckpoint(output_dir, "stage_07_embedding")
        zs, cov_zs, est_contrasts = distributed_stages.distributed_embedding(
            cryos, means, u, s, volume_mask, gpu_memory, options,
            focus_masks, noise_var_used, node_config, ckpt)
        elapsed = time.time() - st_time

        if node_config.rank == 0:
            ckpt.save_config({
                "stage": "embedding", "elapsed_s": elapsed,
                "world_size": node_config.world_size,
            })
            logger.info(f"Stage 'embedding' completed in {elapsed:.1f}s")

    else:
        logger.error(f"Unknown stage: {stage_name}")
        sys.exit(1)

    # Final barrier so all ranks exit together
    barrier(output_dir, f"test_{stage_name}_done", node_config.world_size,
            node_config.job_id, node_config.rank)


def compare_outputs(args):
    """Phase 3: Compare outputs from different node counts."""
    from recovar.stage_checkpoint import StageCheckpoint
    import json

    ref_dir = args.ref_dir
    test_dir = args.test_dir
    stage_name = args.stage

    logger.info(f"Comparing stage '{stage_name}': ref={ref_dir} vs test={test_dir}")

    if stage_name == "mean":
        ref_ckpt = StageCheckpoint(ref_dir, "stage_02_mean")
        test_ckpt = StageCheckpoint(test_dir, "stage_02_mean")

        ref_means = ref_ckpt.load_object("means")
        test_means = test_ckpt.load_object("means")

        ref_prior = ref_ckpt.load_array("mean_prior")
        test_prior = test_ckpt.load_array("mean_prior")

        for key in ['combined', 'corrected0', 'corrected1']:
            if key in ref_means and key in test_means:
                ref_v = np.array(ref_means[key])
                test_v = np.array(test_means[key])
                max_abs = np.max(np.abs(ref_v))
                if max_abs > 0:
                    rel_err = np.max(np.abs(ref_v - test_v)) / max_abs
                else:
                    rel_err = np.max(np.abs(ref_v - test_v))
                status = "PASS" if rel_err < 1e-3 else "FAIL"
                logger.info(f"  means['{key}']: rel_err={rel_err:.2e} [{status}]")

        max_abs = np.max(np.abs(ref_prior))
        if max_abs > 0:
            rel_err = np.max(np.abs(ref_prior - test_prior)) / max_abs
        else:
            rel_err = np.max(np.abs(ref_prior - test_prior))
        status = "PASS" if rel_err < 1e-3 else "FAIL"
        logger.info(f"  mean_prior: rel_err={rel_err:.2e} [{status}]")

    elif stage_name == "noise_variance":
        ref_ckpt = StageCheckpoint(ref_dir, "stage_04_noise")
        test_ckpt = StageCheckpoint(test_dir, "stage_04_noise")

        ref_result = ref_ckpt.load_object("noise_refine_result") if ref_ckpt.has_object("noise_refine_result") else ref_ckpt.load_object("noise_result")
        test_result = test_ckpt.load_object("noise_refine_result") if test_ckpt.has_object("noise_refine_result") else test_ckpt.load_object("noise_result")

        # Compare noise_var_used (index 0) and variance_est (index 1)
        for idx, name in [(0, "noise_var_used")]:
            ref_v = np.array(ref_result[idx])
            test_v = np.array(test_result[idx])
            max_abs = np.max(np.abs(ref_v))
            rel_err = np.max(np.abs(ref_v - test_v)) / max(max_abs, 1e-10)
            status = "PASS" if rel_err < 1e-3 else "FAIL"
            logger.info(f"  {name}: rel_err={rel_err:.2e} [{status}]")

    elif stage_name == "covariance_hb":
        ref_ckpt = StageCheckpoint(ref_dir, "stage_05_covariance")
        test_ckpt = StageCheckpoint(test_dir, "stage_05_covariance")

        ref_result = ref_ckpt.load_object("covariance_hb_result")
        test_result = test_ckpt.load_object("covariance_hb_result")

        ref_Hs, ref_Bs = ref_result
        test_Hs, test_Bs = test_result

        for h in range(2):
            for name, ref_v, test_v in [("H", ref_Hs[h], test_Hs[h]), ("B", ref_Bs[h], test_Bs[h])]:
                ref_v = np.array(ref_v)
                test_v = np.array(test_v)
                max_abs = np.max(np.abs(ref_v))
                rel_err = np.max(np.abs(ref_v - test_v)) / max(max_abs, 1e-10)
                status = "PASS" if rel_err < 1e-3 else "FAIL"
                logger.info(f"  half{h}_{name}: shape={ref_v.shape}, rel_err={rel_err:.2e} [{status}]")

    elif stage_name == "embedding":
        ref_ckpt = StageCheckpoint(ref_dir, "stage_07_embedding")
        test_ckpt = StageCheckpoint(test_dir, "stage_07_embedding")

        ref_result = ref_ckpt.load_object("embedding_result")
        test_result = test_ckpt.load_object("embedding_result")

        ref_zs, ref_cov_zs, ref_contrasts = ref_result
        test_zs, test_cov_zs, test_contrasts = test_result

        for key in ref_zs:
            ref_v = np.array(ref_zs[key])
            test_v = np.array(test_zs[key])
            max_abs = np.max(np.abs(ref_v))
            rel_err = np.max(np.abs(ref_v - test_v)) / max(max_abs, 1e-10)
            status = "PASS" if rel_err < 1e-3 else "FAIL"
            logger.info(f"  zs[{key}]: shape={ref_v.shape}, rel_err={rel_err:.2e} [{status}]")

    else:
        logger.error(f"Unknown stage for comparison: {stage_name}")
        sys.exit(1)

    # Print timing comparison if available
    for label, d in [("ref", ref_dir), ("test", test_dir)]:
        stage_dirs = {
            "mean": "stage_02_mean",
            "noise_variance": "stage_04_noise",
            "covariance_hb": "stage_05_covariance",
            "embedding": "stage_07_embedding",
        }
        ckpt_name = stage_dirs.get(stage_name, stage_name)
        config_path = os.path.join(d, ckpt_name, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            elapsed = config.get("elapsed_s", "?")
            ws = config.get("world_size", "?")
            logger.info(f"  {label}: {elapsed:.1f}s (world_size={ws})" if isinstance(elapsed, float) else f"  {label}: {elapsed}s (world_size={ws})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    # Phase 1: reference
    p_ref = sub.add_parser("reference", help="Generate reference checkpoints")
    p_ref.add_argument("--dataset-dir", required=True, help="Path to test_dataset directory")
    p_ref.add_argument("--image-size", type=int, default=128)

    # Phase 2: run-stage
    p_run = sub.add_parser("run-stage", help="Run a single distributed stage")
    p_run.add_argument("--stage", required=True,
                       choices=["mean", "noise_variance", "covariance_hb", "embedding"])
    p_run.add_argument("--ref-checkpoint", required=True,
                       help="Reference checkpoint dir (from phase 1)")
    p_run.add_argument("--output-dir", required=True,
                       help="Output directory for this test run")

    # Phase 3: compare
    p_cmp = sub.add_parser("compare", help="Compare outputs from different runs")
    p_cmp.add_argument("--ref-dir", required=True)
    p_cmp.add_argument("--test-dir", required=True)
    p_cmp.add_argument("--stage", required=True,
                       choices=["mean", "noise_variance", "covariance_hb", "embedding"])

    args = parser.parse_args()
    if args.command == "reference":
        run_reference(args)
    elif args.command == "run-stage":
        run_stage(args)
    elif args.command == "compare":
        compare_outputs(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
