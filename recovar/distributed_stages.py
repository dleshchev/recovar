"""
Distributed stage wrappers for multi-node RECOVAR computation.

Each wrapper handles:
- Node work assignment (which half, which images/frequencies)
- Partial computation using existing stage functions with image_subset
- Writing partials to shared filesystem
- Barrier synchronization
- Rank 0 reduction (sum or concatenate)
- Broadcasting results to all ranks
"""

import time
import logging
import numpy as np

from recovar.distributed import (
    NodeConfig,
    Assignment,
    FrequencyAssignment,
    compute_image_assignments,
    compute_frequency_assignments,
    write_partial,
    read_and_reduce_partials,
    read_and_concat_partials,
    barrier,
    cleanup_barriers,
    broadcast_value,
)
from recovar.stage_checkpoint import StageCheckpoint
from recovar import stages

logger = logging.getLogger(__name__)


def distributed_mean(cryos, batch_size, noise_var_from_hf, args, node_config, checkpoint):
    """Compute mean conformation across multiple nodes.

    For world_size=1, delegates directly to stages.stage_mean().
    For world_size>=2, each rank computes partial ft_ctf/ft_y for its
    assigned half and image subset using relion_style_triangular_kernel,
    then rank 0 reduces, post-processes, and broadcasts the result.

    Args:
        cryos: List of two cryo datasets (half-sets).
        batch_size: Image batch size for processing.
        noise_var_from_hf: Noise variance estimate (scalar or array).
        args: Pipeline arguments namespace.
        node_config: NodeConfig with rank, world_size, job_id.
        checkpoint: StageCheckpoint for this stage.

    Returns:
        Tuple of (means, mean_prior, uninvert_applied).
        means is a dict with keys 'corrected0', 'corrected1', 'combined', etc.
        uninvert_applied is True if the data sign was swapped.
    """
    if node_config.world_size == 1:
        return stages.stage_mean(cryos, batch_size, noise_var_from_hf, args, st_time=time.time())

    from recovar import relion_functions, regularization
    from recovar.fourier_transform_utils import fourier_transform_utils
    from recovar import utils
    import jax.numpy as jnp
    ftu = fourier_transform_utils(jnp)

    st_time = time.time()
    partial_dir = checkpoint.dir

    # Determine mean_fn parameters
    if args.mean_fn == 'triangular':
        effective_batch_size = 2 * batch_size
        use_regularization = False
    elif args.mean_fn == 'triangular_reg':
        effective_batch_size = 5 * batch_size
        use_regularization = True
    else:
        raise ValueError(f"mean function {args.mean_fn} not recognized")

    upsampling_factor = 2
    disc_type = 'linear_interp'

    # Compute image assignments: each rank gets a half + image subset
    n_images_per_half = [cryos[0].n_images, cryos[1].n_images]
    assignments = compute_image_assignments(node_config.world_size, n_images_per_half)
    my_assignment = assignments[node_config.rank]
    my_half = my_assignment.half
    my_images = my_assignment.image_indices

    logger.info(
        f"Rank {node_config.rank}: computing mean partial for half {my_half}, "
        f"{len(my_images)} images"
    )

    # Set upsampling on local cryo for this half
    cryo = cryos[my_half]
    original_upsampling = cryo.volume_upsampling_factor
    cryo.update_volume_upsampling_factor(upsampling_factor)

    noise_var = noise_var_from_hf.astype(np.float32)

    # Compute partial ft_ctf, ft_y for assigned half and image subset
    ft_ctf_partial, ft_y_partial = relion_functions.relion_style_triangular_kernel(
        cryo, noise_var, effective_batch_size, disc_type=disc_type,
        image_subset=my_images
    )

    # Restore upsampling
    cryo.update_volume_upsampling_factor(original_upsampling)

    # Convert to numpy for writing
    ft_ctf_partial = np.array(ft_ctf_partial)
    ft_y_partial = np.array(ft_y_partial)

    # Write partials
    write_partial(
        checkpoint.partial_path(node_config.rank, f"half{my_half}_ft_ctf"),
        ft_ctf_partial
    )
    write_partial(
        checkpoint.partial_path(node_config.rank, f"half{my_half}_ft_y"),
        ft_y_partial
    )

    logger.info(f"Rank {node_config.rank}: wrote mean partials in {time.time() - st_time:.1f}s")

    # Barrier: wait for all ranks to finish writing partials
    barrier(partial_dir, "mean_partials", node_config.world_size,
            node_config.job_id, node_config.rank)

    # Rank 0: reduce partials and post-process
    if node_config.rank == 0:
        logger.info("Rank 0: reducing mean partials and post-processing")

        means = {}
        ft_ctfs = [None, None]
        ft_ys = [None, None]

        for h in range(2):
            ft_ctfs[h] = read_and_reduce_partials(partial_dir, f"partial_*_half{h}_ft_ctf.npy")
            ft_ys[h] = read_and_reduce_partials(partial_dir, f"partial_*_half{h}_ft_y.npy")

        # Set upsampling for post-processing
        original_upsamplings = []
        for idx, cryo_h in enumerate(cryos):
            original_upsamplings.append(cryo_h.volume_upsampling_factor)
            cryo_h.update_volume_upsampling_factor(upsampling_factor)

        # Post-process: unregularized reconstructions
        for idx, cryo_h in enumerate(cryos):
            means[f"corrected{idx}"] = relion_functions.post_process_from_filter(
                cryo_h, ft_ctfs[idx], ft_ys[idx], tau=None, disc_type=disc_type,
                use_spherical_mask=True, grid_correct=True,
                gridding_correct="square", kernel_width=1
            )

        # Compute prior from unregularized reconstructions
        mean_prior, fsc, _ = regularization.compute_relion_prior(
            cryos, noise_var, means["corrected0"], means["corrected1"], effective_batch_size
        )

        # Combined mean
        means["combined"] = (means["corrected0"] + means["corrected1"]) / 2

        # Regularized reconstructions
        for idx, cryo_h in enumerate(cryos):
            means[f"corrected{idx}reg"] = relion_functions.post_process_from_filter(
                cryo_h, ft_ctfs[idx], ft_ys[idx], tau=mean_prior, disc_type=disc_type,
                use_spherical_mask=True, grid_correct=True,
                gridding_correct="square", kernel_width=1
            )
            cryo_h.update_volume_upsampling_factor(original_upsamplings[idx])

        means["combined_regularized"] = (means["corrected0reg"] + means["corrected1reg"]) / 2

        if use_regularization:
            means["combined"] = means["combined_regularized"]

        # Additional results
        lhs = (ft_ctfs[0] + ft_ctfs[1]) / 2
        mean_prior = np.array(mean_prior)
        means["prior"] = mean_prior
        means["lhs"] = lhs

        # Convert all to numpy
        for key in means:
            means[key] = np.array(means[key])

        # Uninvert check (same logic as stage_mean / get_mean_conformation_relion)
        mean_real = ftu.get_idft3(means['combined'].reshape(cryos[0].volume_shape))
        uninvert_check = np.sum(
            (mean_real.real ** 3
             * cryos[0].get_volume_radial_mask(cryos[0].grid_size // 3).reshape(cryos[0].volume_shape))
        ) < 0

        uninvert_applied = False
        if args.uninvert_data == 'automatic':
            if uninvert_check:
                for key in ['combined', 'init0', 'init1', 'corrected0', 'corrected1']:
                    if key in means:
                        means[key] = -means[key]
                args.uninvert_data = "true"
                logger.warning('sum(mean) < 0! swapping sign of data (uninvert-data = true)')
                uninvert_applied = True
            else:
                logger.info('setting (uninvert-data = false)')
                args.uninvert_data = "false"
        elif uninvert_check:
            logger.warning(
                'sum(mean) < 0! Data probably needs to be inverted! '
                'set --uninvert-data=true (or automatic)'
            )

        if means['combined'].dtype != cryos[0].dtype:
            logger.warning(f"mean estimate is in type: {means['combined'].dtype}")
            means['combined'] = means['combined'].astype(cryos[0].dtype)

        # Save results for other ranks
        checkpoint.save_object("means", means)
        checkpoint.save_array("mean_prior", mean_prior)
        checkpoint.save_config({"uninvert_applied": uninvert_applied})
        checkpoint.mark_complete()

        logger.info(f"Rank 0: mean post-processing completed in {time.time() - st_time:.1f}s")
    else:
        # Restore upsampling on non-rank-0 nodes (they set it above for their half)
        # Other halves were never changed, so this is safe.
        pass

    # Barrier: wait for rank 0 to finish post-processing
    barrier(partial_dir, "mean_complete", node_config.world_size,
            node_config.job_id, node_config.rank)

    # All ranks load results
    means = checkpoint.load_object("means")
    mean_prior = checkpoint.load_array("mean_prior")
    config = checkpoint.load_config()
    uninvert_applied = config["uninvert_applied"]

    # Apply uninvert to local cryos if needed
    if uninvert_applied:
        for cryo_h in cryos:
            cryo_h.image_stack.mult = -1 * cryo_h.image_stack.mult
        logger.info(f"Rank {node_config.rank}: applied uninvert to local cryos")

    utils.report_memory_device(logger=logger)
    logger.info(f"Rank {node_config.rank}: mean stage completed in {time.time() - st_time:.1f}s")

    return means, mean_prior, uninvert_applied


def distributed_mask(args, means, volume_shape, dtype_real, cryos, node_config, checkpoint):
    """Compute masks and save mean/mask volumes to disk.

    Runs on rank 0 only since mask computation is fast and does not
    iterate over images. Other ranks wait at barrier and load the result.

    Args:
        args: Pipeline arguments namespace.
        means: Dict of mean volumes from distributed_mean.
        volume_shape: 3-tuple of volume dimensions.
        dtype_real: Real-valued dtype (e.g., np.float32).
        cryos: List of two cryo datasets.
        node_config: NodeConfig with rank, world_size, job_id.
        checkpoint: StageCheckpoint for this stage.

    Returns:
        Tuple of (volume_mask, dilated_volume_mask, focus_masks).
    """
    if node_config.world_size == 1:
        return stages.stage_mask(args, means, volume_shape, dtype_real, cryos)

    if node_config.rank == 0:
        result = stages.stage_mask(args, means, volume_shape, dtype_real, cryos)
        checkpoint.save_object("mask_result", result)
        checkpoint.mark_complete()
        logger.info("Rank 0: mask computation complete")

    barrier(checkpoint.dir, "mask", node_config.world_size,
            node_config.job_id, node_config.rank)

    result = checkpoint.load_object("mask_result")
    logger.info(f"Rank {node_config.rank}: loaded mask result")
    return result


def distributed_noise_refine_and_variance(cryo, cryos, means, batch_size,
                                          dilated_volume_mask, args, noise_model,
                                          node_config, checkpoint):
    """Refine noise estimates and compute variance.

    Runs on rank 0 only for the initial implementation. The noise estimation
    functions have complex internals that operate primarily on cryos[0] and
    are not easily parallelized. Parallelizing the final compute_variance
    call is deferred to a later optimization.

    Args:
        cryo: Primary cryo dataset (cryos[0]).
        cryos: List of two cryo datasets.
        means: Dict of mean volumes.
        batch_size: Image batch size.
        dilated_volume_mask: Dilated volume mask array.
        args: Pipeline arguments namespace.
        noise_model: Noise model type string ('radial' or 'radial_per_tilt').
        node_config: NodeConfig with rank, world_size, job_id.
        checkpoint: StageCheckpoint for this stage.

    Returns:
        Tuple of (noise_var_used, variance_est, variance_fsc, noise_p_variance_est,
                  radial_noise_var_outside_mask, radial_ub_noise_var,
                  white_noise_var_outside_mask, image_PS, std_image_PS,
                  masked_image_PS, std_masked_image_PS, ub_noise_var_by_var_est).
    """
    from recovar import noise as noise_module

    if node_config.world_size == 1:
        return stages.stage_noise_refine_and_variance(
            cryo, cryos, means, batch_size, dilated_volume_mask, args, noise_model
        )

    if node_config.rank == 0:
        logger.info("Rank 0: running noise refinement and variance computation")
        result = stages.stage_noise_refine_and_variance(
            cryo, cryos, means, batch_size, dilated_volume_mask, args, noise_model
        )
        noise_var_used = result[0]
        checkpoint.save_array("noise_var_used", noise_var_used)
        checkpoint.save_object("noise_refine_result", result)
        checkpoint.mark_complete()
        logger.info("Rank 0: noise refinement complete")

    barrier(checkpoint.dir, "noise_refine", node_config.world_size,
            node_config.job_id, node_config.rank)

    # All ranks load results
    result = checkpoint.load_object("noise_refine_result")
    noise_var_used = result[0]

    # All ranks update their local cryos noise model
    noise_module.update_noise_variance(noise_var_used, cryos)
    logger.info(f"Rank {node_config.rank}: updated local noise model")

    return result


def distributed_covariance_hb(cryos, means, dilated_volume_mask, picked_frequencies,
                              gpu_memory, options, node_config, checkpoint):
    """Compute covariance H and B matrices across multiple nodes.

    Frequency-split across nodes:
    - world_size=1: calls compute_both_H_B directly.
    - world_size=2: rank 0 handles half 0, rank 1 handles half 1.
    - world_size>=4: 2 halves x N/2 frequency ranges per half.

    Each rank calls compute_H_B_in_volume_batch for its assigned half and
    frequency subset. Rank 0 concatenates frequency ranges along axis=1
    for each half.

    Args:
        cryos: List of two cryo datasets.
        means: Dict of mean volumes.
        dilated_volume_mask: Dilated volume mask array.
        picked_frequencies: Array of picked frequency indices.
        gpu_memory: Available GPU memory in bytes.
        options: Covariance computation options dict.
        node_config: NodeConfig with rank, world_size, job_id.
        checkpoint: StageCheckpoint for this stage.

    Returns:
        Tuple of (Hs, Bs) where Hs = [H_half0, H_half1] and
        Bs = [B_half0, B_half1].
    """
    from recovar import covariance_estimation

    if node_config.world_size == 1:
        return covariance_estimation.compute_both_H_B(
            cryos, means, dilated_volume_mask, picked_frequencies,
            gpu_memory, parallel_analysis=False, options=options
        )

    st_time = time.time()
    n_frequencies = len(picked_frequencies)
    freq_assignments = compute_frequency_assignments(node_config.world_size, n_frequencies)
    my_freq = freq_assignments[node_config.rank]
    my_half = my_freq.half
    my_freq_start = my_freq.freq_start
    my_freq_end = my_freq.freq_end
    my_picked_frequencies = picked_frequencies[my_freq_start:my_freq_end]

    logger.info(
        f"Rank {node_config.rank}: computing H/B for half {my_half}, "
        f"frequencies [{my_freq_start}:{my_freq_end}] "
        f"({len(my_picked_frequencies)} of {n_frequencies})"
    )

    # Select mean for this half
    mean = means["combined"] if options["use_combined_mean"] else means[f"corrected{my_half}"]

    # Compute H, B for assigned half and frequency subset
    H, B = covariance_estimation.compute_H_B_in_volume_batch(
        cryos[my_half], mean, dilated_volume_mask, my_picked_frequencies,
        gpu_memory, parallel_analysis=False, options=options
    )

    H = np.array(H)
    B = np.array(B)

    # Write partials
    write_partial(
        checkpoint.partial_path(node_config.rank, f"half{my_half}_H"),
        H
    )
    write_partial(
        checkpoint.partial_path(node_config.rank, f"half{my_half}_B"),
        B
    )

    logger.info(
        f"Rank {node_config.rank}: wrote H/B partials "
        f"(H shape={H.shape}, B shape={B.shape}) "
        f"in {time.time() - st_time:.1f}s"
    )

    # Free GPU memory
    del H, B

    # Barrier: wait for all ranks to finish writing
    barrier(checkpoint.dir, "covariance_hb_partials", node_config.world_size,
            node_config.job_id, node_config.rank)

    # Rank 0: assemble full H, B per half
    if node_config.rank == 0:
        logger.info("Rank 0: assembling covariance H/B matrices")
        Hs = []
        Bs = []

        for h in range(2):
            if node_config.world_size == 2:
                # Each rank has all frequencies for one half, no concatenation needed
                H_h = np.load(
                    checkpoint.partial_path(h, f"half{h}_H")
                )
                B_h = np.load(
                    checkpoint.partial_path(h, f"half{h}_B")
                )
            else:
                # Multiple ranks per half: concatenate along frequency axis (axis=1)
                H_h = read_and_concat_partials(
                    checkpoint.dir, f"partial_*_half{h}_H.npy", axis=1
                )
                B_h = read_and_concat_partials(
                    checkpoint.dir, f"partial_*_half{h}_B.npy", axis=1
                )

            Hs.append(H_h)
            Bs.append(B_h)

        checkpoint.save_object("covariance_hb_result", (Hs, Bs))
        checkpoint.mark_complete()
        logger.info(
            f"Rank 0: covariance H/B assembly complete "
            f"(H shapes: {[h.shape for h in Hs]}) "
            f"in {time.time() - st_time:.1f}s"
        )

    # Barrier: wait for rank 0 to finish assembly
    barrier(checkpoint.dir, "covariance_hb_complete", node_config.world_size,
            node_config.job_id, node_config.rank)

    # All ranks load results
    Hs, Bs = checkpoint.load_object("covariance_hb_result")
    logger.info(f"Rank {node_config.rank}: loaded covariance H/B result")

    return Hs, Bs


def distributed_covariance_pca(cryos, options, means, mean_prior, focus_masks,
                               dilated_volume_mask, valid_idx, batch_size,
                               gpu_memory, variance_est, args,
                               node_config, checkpoint):
    """Compute covariance regularization and PCA.

    Runs on rank 0 only for the initial implementation. The regularization
    and SVD steps are not easily parallelized and are much cheaper than the
    H/B computation (which is handled by distributed_covariance_hb).

    In a future version, the H/B computation within stage_covariance_pca
    will be replaced with distributed_covariance_hb + rank-0 regularization.

    Args:
        cryos: List of two cryo datasets.
        options: Algorithm options dict.
        means: Dict of mean volumes.
        mean_prior: Mean prior array.
        focus_masks: List of focus mask arrays.
        dilated_volume_mask: Dilated volume mask array.
        valid_idx: Valid frequency indices.
        batch_size: Image batch size.
        gpu_memory: Available GPU memory in bytes.
        variance_est: Variance estimate dict.
        args: Pipeline arguments namespace.
        node_config: NodeConfig with rank, world_size, job_id.
        checkpoint: StageCheckpoint for this stage.

    Returns:
        Tuple of (u, s, covariance_cols, picked_frequencies, column_fscs,
                  covariance_options).
    """
    if node_config.world_size == 1:
        return stages.stage_covariance_pca(
            cryos, options, means, mean_prior, focus_masks,
            dilated_volume_mask, valid_idx, batch_size, gpu_memory,
            variance_est, args
        )

    if node_config.rank == 0:
        logger.info("Rank 0: running covariance PCA")
        result = stages.stage_covariance_pca(
            cryos, options, means, mean_prior, focus_masks,
            dilated_volume_mask, valid_idx, batch_size, gpu_memory,
            variance_est, args
        )
        checkpoint.save_object("covariance_pca_result", result)
        checkpoint.mark_complete()
        logger.info("Rank 0: covariance PCA complete")

    barrier(checkpoint.dir, "covariance_pca", node_config.world_size,
            node_config.job_id, node_config.rank)

    result = checkpoint.load_object("covariance_pca_result")
    logger.info(f"Rank {node_config.rank}: loaded covariance PCA result")
    return result


def distributed_embedding(cryos, means, u, s, volume_mask, gpu_memory, options,
                          focus_masks, noise_var_used, node_config, checkpoint):
    """Compute per-image embeddings for each zdim.

    Runs on rank 0 only for the initial implementation. Embedding is
    relatively fast compared to covariance computation, and the internal
    per-half iteration in get_per_image_embedding makes it complex to
    split across nodes without refactoring.

    Args:
        cryos: List of two cryo datasets.
        means: Dict of mean volumes.
        u: Dict with 'rescaled' key containing principal component vectors.
        s: Dict with 'rescaled' key containing singular values.
        volume_mask: Volume mask array.
        gpu_memory: Available GPU memory in bytes.
        options: Algorithm options dict.
        focus_masks: List of focus mask arrays.
        noise_var_used: Noise variance array used for embedding.
        node_config: NodeConfig with rank, world_size, job_id.
        checkpoint: StageCheckpoint for this stage.

    Returns:
        Tuple of (zs, cov_zs, est_contrasts).
    """
    if node_config.world_size == 1:
        return stages.stage_embedding(
            cryos, means, u, s, volume_mask, gpu_memory, options,
            focus_masks, noise_var_used
        )

    if node_config.rank == 0:
        logger.info("Rank 0: running embedding computation")
        result = stages.stage_embedding(
            cryos, means, u, s, volume_mask, gpu_memory, options,
            focus_masks, noise_var_used
        )
        checkpoint.save_object("embedding_result", result)
        checkpoint.mark_complete()
        logger.info("Rank 0: embedding computation complete")

    barrier(checkpoint.dir, "embedding", node_config.world_size,
            node_config.job_id, node_config.rank)

    result = checkpoint.load_object("embedding_result")
    logger.info(f"Rank {node_config.rank}: loaded embedding result")
    return result
