"""
Distributed multi-node pipeline for RECOVAR.

Invoked as: recovar pipeline_distributed <args>

Same pipeline as `recovar pipeline` but with multi-node support via
file-based coordination on a shared filesystem. Each SLURM task runs
this command; nodes coordinate via barrier files and partial results.

For single-node (world_size=1), behavior is identical to the standard pipeline.
"""

import logging
logger = logging.getLogger(__name__)
import recovar.config
import jax
import jax.numpy as jnp
import numpy as np
import os, argparse, time, pickle, sys

from recovar import dataset, utils, noise, embedding
from recovar import covariance_estimation
from recovar import output as o
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

from recovar.distributed import (
    NodeConfig, get_node_config_from_env, barrier, broadcast_value
)
from recovar.stage_checkpoint import StageCheckpoint
from recovar import stages
from recovar import distributed_stages


def add_args(parser: argparse.ArgumentParser):
    """Add pipeline_distributed arguments. Inherits all pipeline args plus distributed-specific ones."""
    # Import and reuse pipeline's add_args
    from recovar.commands.pipeline import add_args as pipeline_add_args
    pipeline_add_args(parser)

    # Distributed-specific arguments
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default=None,
        type=os.path.abspath,
        help="Checkpoint directory for stage results. "
             "Default: {outdir}/checkpoint. "
             "Override with RECOVAR_CHECKPOINT_DIR env var."
    )
    parser.add_argument(
        "--resume-from-stage",
        dest="resume_from_stage",
        default=None,
        type=int,
        help="Resume from stage N (skip stages 0..N-1). "
             "Requires their DONE markers to exist in checkpoint dir."
    )
    parser.add_argument(
        "--keep-checkpoints",
        dest="keep_checkpoints",
        action="store_true",
        default=False,
        help="Keep checkpoint directories after successful completion. "
             "Default: delete to save disk space."
    )
    return parser


def resolve_checkpoint_dir(args):
    """Determine checkpoint directory from CLI > env var > default."""
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    env_dir = os.environ.get("RECOVAR_CHECKPOINT_DIR")
    if env_dir is not None:
        return os.path.abspath(env_dir)
    return os.path.join(args.outdir, "checkpoint")


def distributed_recovar_pipeline(args):
    """Run the RECOVAR pipeline with multi-node distributed support."""
    st_time = time.time()

    # Get node identity
    node_config = get_node_config_from_env()
    checkpoint_dir = resolve_checkpoint_dir(args)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(
        f"Distributed pipeline starting: rank={node_config.rank}, "
        f"world_size={node_config.world_size}, "
        f"checkpoint_dir={checkpoint_dir}"
    )

    # --- Stage 0: Setup (rank 0 runs, results broadcast) ---
    ckpt_setup = StageCheckpoint(checkpoint_dir, "stage_00_setup")
    if not ckpt_setup.is_complete():
        if node_config.rank == 0:
            (cryos, ind_split, options, batch_size, gpu_memory, noise_var_from_hf,
             valid_idx, noise_model, n_repeats, path_mapping,
             dataset_loader_dict) = stages.stage_setup(args)

            ckpt_setup.save_object("setup_result", {
                'ind_split': ind_split,
                'options': options,
                'batch_size': batch_size,
                'gpu_memory': gpu_memory,
                'noise_var_from_hf': noise_var_from_hf,
                'valid_idx': valid_idx,
                'noise_model': noise_model,
                'n_repeats': n_repeats,
                'path_mapping': path_mapping,
                'dataset_loader_dict': dataset_loader_dict,
            })
            ckpt_setup.mark_complete()

        barrier(checkpoint_dir, "setup", node_config.world_size,
                node_config.job_id, node_config.rank)

    # All ranks load setup results and reconstruct cryos
    setup = ckpt_setup.load_object("setup_result")
    ind_split = setup['ind_split']
    options = setup['options']
    batch_size = setup['batch_size']
    gpu_memory = setup['gpu_memory']
    noise_var_from_hf = setup['noise_var_from_hf']
    valid_idx = setup['valid_idx']
    noise_model = setup['noise_model']
    n_repeats = setup['n_repeats']
    path_mapping = setup['path_mapping']
    dataset_loader_dict = setup['dataset_loader_dict']

    # All ranks load datasets independently
    cryos = dataset.get_split_datasets_from_dict(dataset_loader_dict, ind_split, args.lazy)

    # Initialize noise model on all ranks
    for cryo in cryos:
        if noise_model == "radial":
            cryo.set_radial_noise_model(None)
        elif noise_model in ('radial_per_tilt', 'radial-per-tilt'):
            cryo.set_variable_radial_noise_model(None)
        else:
            raise ValueError(f"noise model {noise_model} not recognized")

    logger.info(f"Rank {node_config.rank}: datasets loaded, {cryos[0].n_images + cryos[1].n_images} images total")

    # --- Per-pass loop ---
    contrasts_for_second = None
    for repeat in range(n_repeats):
        pass_suffix = f"_pass{repeat}" if n_repeats > 1 else ""

        # Apply contrasts from previous pass
        if repeat == 1:
            contrasts_for_second = stages.apply_contrast_correction(
                cryos, options, est_contrasts)

        # --- Stage 2: Mean ---
        ckpt_mean = StageCheckpoint(checkpoint_dir, f"stage_02_mean{pass_suffix}")
        if not ckpt_mean.is_complete():
            means, mean_prior, uninvert_applied = distributed_stages.distributed_mean(
                cryos, batch_size, noise_var_from_hf, args,
                node_config, ckpt_mean)
        else:
            means = ckpt_mean.load_object("means")
            mean_prior = ckpt_mean.load_array("mean_prior")
            uninvert_applied = ckpt_mean.load_config().get("uninvert_applied", False)
            if uninvert_applied:
                for cryo in cryos:
                    cryo.image_stack.mult = -1 * cryo.image_stack.mult
            logger.info(f"Rank {node_config.rank}: skipping mean (already complete)")

        # --- Stage 3: Mask ---
        ckpt_mask = StageCheckpoint(checkpoint_dir, f"stage_03_mask{pass_suffix}")
        if not ckpt_mask.is_complete():
            volume_mask, dilated_volume_mask, focus_masks = distributed_stages.distributed_mask(
                args, means, cryos[0].volume_shape, cryos[0].dtype_real, cryos,
                node_config, ckpt_mask)
        else:
            result = ckpt_mask.load_object("mask_result")
            volume_mask, dilated_volume_mask, focus_masks = result
            logger.info(f"Rank {node_config.rank}: skipping mask (already complete)")

        if args.only_mean:
            return

        # Check mask dtype
        if volume_mask.dtype != np.float32:
            raise TypeError(f"volume_mask is not of dtype float32, but {volume_mask.dtype}")

        # --- Stage 4: Noise refinement + Variance ---
        ckpt_noise = StageCheckpoint(checkpoint_dir, f"stage_04_noise{pass_suffix}")
        if not ckpt_noise.is_complete():
            (noise_var_used, variance_est, variance_fsc, noise_p_variance_est,
             radial_noise_var_outside_mask, radial_ub_noise_var,
             white_noise_var_outside_mask, image_PS, std_image_PS,
             masked_image_PS, std_masked_image_PS,
             ub_noise_var_by_var_est) = distributed_stages.distributed_noise_refine_and_variance(
                cryos[0], cryos, means, batch_size, dilated_volume_mask, args, noise_model,
                node_config, ckpt_noise)
        else:
            result = ckpt_noise.load_object("noise_refine_result")
            (noise_var_used, variance_est, variance_fsc, noise_p_variance_est,
             radial_noise_var_outside_mask, radial_ub_noise_var,
             white_noise_var_outside_mask, image_PS, std_image_PS,
             masked_image_PS, std_masked_image_PS,
             ub_noise_var_by_var_est) = result
            noise.update_noise_variance(noise_var_used, cryos)
            logger.info(f"Rank {node_config.rank}: skipping noise/variance (already complete)")

        # --- Stage 5+6: Covariance + PCA ---
        ckpt_pca = StageCheckpoint(checkpoint_dir, f"stage_06_pca{pass_suffix}")
        if not ckpt_pca.is_complete():
            (u, s, covariance_cols, picked_frequencies, column_fscs,
             covariance_options) = distributed_stages.distributed_covariance_pca(
                cryos, options, means, mean_prior, focus_masks,
                dilated_volume_mask, valid_idx, batch_size, gpu_memory,
                variance_est, args, node_config, ckpt_pca)
        else:
            result = ckpt_pca.load_object("covariance_pca_result")
            (u, s, covariance_cols, picked_frequencies, column_fscs,
             covariance_options) = result
            logger.info(f"Rank {node_config.rank}: skipping covariance/PCA (already complete)")

        # --- Stage 7: Embedding ---
        ckpt_embed = StageCheckpoint(checkpoint_dir, f"stage_07_embedding{pass_suffix}")
        if not ckpt_embed.is_complete():
            zs, cov_zs, est_contrasts = distributed_stages.distributed_embedding(
                cryos, means, u, s, volume_mask, gpu_memory, options,
                focus_masks, noise_var_used, node_config, ckpt_embed)
        else:
            result = ckpt_embed.load_object("embedding_result")
            zs, cov_zs, est_contrasts = result
            logger.info(f"Rank {node_config.rank}: skipping embedding (already complete)")

        # Contrast rescaling for second pass
        if repeat == 1:
            for key in est_contrasts:
                est_contrasts[key] = est_contrasts[key] * contrasts_for_second

    # --- Stage 8: Save (rank 0 only) ---
    if node_config.rank == 0:
        stages.stage_save(args, cryos, means, u, s, volume_mask,
                          dilated_volume_mask, focus_masks, zs, cov_zs,
                          est_contrasts, noise_var_from_hf, noise_var_used,
                          None,  # noise_var_from_het_residual computed inside
                          radial_noise_var_outside_mask, radial_ub_noise_var,
                          white_noise_var_outside_mask, ub_noise_var_by_var_est,
                          image_PS, std_image_PS, masked_image_PS,
                          std_masked_image_PS, variance_est, variance_fsc,
                          noise_p_variance_est, covariance_cols, covariance_options,
                          column_fscs, picked_frequencies, contrasts_for_second,
                          options, ind_split, path_mapping, st_time)

    # Final barrier so all ranks exit together
    barrier(checkpoint_dir, "pipeline_complete", node_config.world_size,
            node_config.job_id, node_config.rank)

    # Cleanup checkpoints
    if node_config.rank == 0 and not args.keep_checkpoints:
        import shutil
        logger.info(f"Cleaning up checkpoint directory: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    total_time = time.time() - st_time
    logger.info(f"Rank {node_config.rank}: distributed pipeline completed in {total_time:.1f}s")

    if node_config.rank == 0:
        return means, u, s, volume_mask, dilated_volume_mask, noise_var_used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    distributed_recovar_pipeline(args)


if __name__ == "__main__":
    main()
