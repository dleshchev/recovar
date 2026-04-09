"""
Pipeline stage functions extracted from standard_recovar_pipeline.

Each function corresponds to a discrete pipeline stage that can be
checkpointed and distributed independently.
"""

import time
import logging
import numpy as np
import pickle
import sys
import os

from recovar import dataset, homogeneous, embedding, principal_components
from recovar import covariance_estimation, noise, mask, utils, constants
from recovar import output as o
from recovar.fourier_transform_utils import fourier_transform_utils
from recovar.utils_core import copy_data_to_temp_folder, save_original_paths_info, cleanup_temp_files

import jax.numpy as jnp
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)


def stage_setup(args):
    """
    Pre-loop setup: args processing, dataset loading, batch sizes, initial noise,
    noise model initialization, contrast repeat setup.

    Corresponds to pipeline.py lines 320-433.

    Returns:
        (cryos, ind_split, options, batch_size, gpu_memory, noise_var_from_hf,
         valid_idx, noise_model, n_repeats, path_mapping, dataset_loader_dict)
    """
    # Copy data to temporary folder if requested
    path_mapping = copy_data_to_temp_folder(args)

    if args.mask.endswith(".mrc"):
        args.mask = os.path.abspath(args.mask)

    if (not args.accept_cpu) and (not utils.jax_has_gpu()):
        raise ValueError("No GPU found. Set --accept-cpu if you really want to run on CPU (probably not). More likely, you want to check that JAX has been properly installed with GPU support.")

    # Dump input arguments
    o.mkdir_safe(args.outdir)
    with open(f"{args.outdir}/command.txt", "w") as text_file:
        command = 'python ' + ' '.join((sys.argv))
        text_file.write(command)

    # Save original paths if data was copied
    save_original_paths_info(path_mapping, args.outdir)

    # Set CTF function here
    if args.tilt_series_ctf is None and args.tilt_series is False:
        args.tilt_series_ctf = 'cryoem'
        logger.info("Setting tilt_series_ctf to cryoem")
    elif args.tilt_series_ctf is None and args.tilt_series is True:
        args.tilt_series_ctf = 'relion5'
        logger.info("Setting tilt_series_ctf to relion5")

    if args.tilt_series and args.dose_per_tilt is not None:
        logger.warning("dose_per_tilt is provided, but tilt_series_ctf is set to using starfile = -B_fac/4 (by default). Thus, dose_per_tilt will not be used.")

    if (args.tilt_series_ctf == 'v2_scale_from_star') and (args.angle_per_tilt is not None):
        logger.warning("angle_per_tilt is provided, but tilt_series_ctf is set to using scale from inputfile (by default). Thus, angle_per_tilt will not be used.")

    if args.do_over_with_contrast is None:
        args.do_over_with_contrast = args.correct_contrast

    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        level=logging.INFO,
                        force = True,
                        handlers=[
        logging.FileHandler(f"{args.outdir}/run.log"),
        logging.StreamHandler()])

    logger.info(args)
    ind_split = dataset.figure_out_halfsets(args)
    dataset_loader_dict = dataset.make_dataset_loader_dict(args)

    # Log the paths being used for data loading
    if path_mapping is not None:
        logger.info("Using copied data files for loading:")
        logger.info(f"  Particles file: {dataset_loader_dict['particles_file']}")
        logger.info(f"  Poses file: {dataset_loader_dict['poses_file']}")
        logger.info(f"  CTF file: {dataset_loader_dict['ctf_file']}")
        if dataset_loader_dict['datadir']:
            logger.info(f"  Datadir: {dataset_loader_dict['datadir']}")
    else:
        logger.info("Using original data files for loading:")
        logger.info(f"  Particles file: {dataset_loader_dict['particles_file']}")
        logger.info(f"  Poses file: {dataset_loader_dict['poses_file']}")
        logger.info(f"  CTF file: {dataset_loader_dict['ctf_file']}")
        if dataset_loader_dict['datadir']:
            logger.info(f"  Datadir: {dataset_loader_dict['datadir']}")

    options = utils.make_algorithm_options(args)

    cryos = dataset.get_split_datasets_from_dict(dataset_loader_dict, ind_split, args.lazy)
    cryo = cryos[0]
    if args.gpu_memory is not None:
        utils.GPU_MEMORY_LIMIT = args.gpu_memory
    gpu_memory = utils.get_gpu_memory_total()
    volume_shape = cryo.volume_shape
    disc_type = "linear_interp"

    batch_size = utils.get_image_batch_size(cryo.grid_size, gpu_memory)
    logger.info(f"image batch size: {batch_size}")
    logger.info(f"volume batch size: {utils.get_vol_batch_size(cryo.grid_size, gpu_memory)}")
    logger.info(f"column batch size: {utils.get_column_batch_size(cryo.grid_size, gpu_memory)}")
    logger.info(f"number of images: {cryos[0].n_images + cryos[1].n_images}")
    utils.report_memory_device(logger=logger)

    noise_var_from_hf, _ = noise.estimate_noise_variance(cryos[0], batch_size)

    valid_idx = cryo.get_valid_frequency_indices()
    noise_model = args.noise_model


    if args.do_over_with_contrast:
        n_repeats = 2
        if not args.correct_contrast:
            logger.warning("Do over with contrast, but contrast correction is off. Setting contrast correction to on")
            args.correct_contrast = True
            options["contrast"] = "contrast_qr"

    else:
        n_repeats = 1

    if args.shared_contrast_across_tilts:
        options['contrast'] += '_shared'
        logger.info("Setting contrast to shared")

    ## Initialize noise model
    for cryo in cryos:
        if noise_model == "radial":
            cryo.set_radial_noise_model(None)
            logger.info("Setting noise model to radial")
        elif noise_model == 'radial_per_tilt' or noise_model == 'radial-per-tilt':
            cryo.set_variable_radial_noise_model(None)
            logger.info("Setting noise model to radial_per_tilt")
        else:
            raise ValueError(f"noise model {noise_model} not recognized")

    return (cryos, ind_split, options, batch_size, gpu_memory, noise_var_from_hf,
            valid_idx, noise_model, n_repeats, path_mapping, dataset_loader_dict)


def apply_contrast_correction(cryos, options, est_contrasts):
    """
    Apply contrast correction at the start of repeat==1.

    Corresponds to pipeline.py lines 436-447.

    Returns:
        contrasts_for_second: normalized contrast array
    """
    if 10 in options['zs_dim_to_test']:
        ndim = 10
    else:
        ndim = np.median(options['zs_dim_to_test'])
    logger.warning(f"repeating with contrast of zdim={ndim}")
    contrasts_for_second = est_contrasts[ndim]
    contrasts_for_second /= np.mean(contrasts_for_second) # normalize to have mean 1
    embedding.set_contrasts_in_cryos(cryos, contrasts_for_second)
    options["contrast"] = "contrast"
    return contrasts_for_second


def stage_mean(cryos, batch_size, noise_var_from_hf, args, st_time=None):
    """
    Compute mean conformation and handle uninvert data check.

    Corresponds to pipeline.py lines 449-487.

    Returns:
        (means, mean_prior, uninvert_applied)
        uninvert_applied is True if the data sign was swapped.
    """
    # Compute mean
    if args.mean_fn == 'triangular':
        means, mean_prior, _  = homogeneous.get_mean_conformation_relion(cryos, 2*batch_size, noise_variance = noise_var_from_hf,  use_regularization = False)
    elif args.mean_fn == 'triangular_reg':
        means, mean_prior, _  = homogeneous.get_mean_conformation_relion(cryos, 5*batch_size, noise_variance = noise_var_from_hf,  use_regularization = True)
    else:
        raise ValueError(f"mean function {args.mean_fn} not recognized")
    utils.report_memory_device(logger=logger)


    mean_real = ftu.get_idft3(means['combined'].reshape(cryos[0].volume_shape))

    ## DECIDE IF WE SHOULD UNINVERT DATA
    uninvert_check = np.sum((mean_real.real**3 * cryos[0].get_volume_radial_mask(cryos[0].grid_size//3).reshape(cryos[0].volume_shape))) < 0
    uninvert_applied = False
    if args.uninvert_data == 'automatic':
        # Check if in real space, things towards the middle are mostly positive or negative
        if uninvert_check:
        # if np.sum(mean_real.real**3 * cryos[0].get_volume_mask() ) < 0:
            for key in ['combined', 'init0', 'init1', 'corrected0', 'corrected1']:
                if key in means:
                    means[key] =- means[key]
            for cryo in cryos:
                cryo.image_stack.mult = -1 * cryo.image_stack.mult
            args.uninvert_data = "true"
            logger.warning('sum(mean) < 0! swapping sign of data (uninvert-data = true)')
            uninvert_applied = True
        else:
            logger.info('setting (uninvert-data = false)')
            args.uninvert_data = "false"
    elif uninvert_check:
        logger.warning('sum(mean) < 0! Data probably needs to be inverted! set --uninvert-data=true (or automatic)')
    ## END OF THIS - maybe move this block of code somewhere else?


    if means['combined'].dtype != cryos[0].dtype:
        logger.warning(f"mean estimate is in type: {means['combined'].dtype}")
        means['combined'] = means['combined'].astype(cryos[0].dtype)

    logger.info(f"mean computed in {time.time() - st_time}" if st_time is not None else "mean computed")
    utils.report_memory_device(logger=logger)

    return means, mean_prior, uninvert_applied


def stage_mask(args, means, volume_shape, dtype_real, cryos):
    """
    Compute masks and save mean/mask volumes to disk.

    Corresponds to pipeline.py lines 489-524.

    Returns:
        (volume_mask, dilated_volume_mask, focus_masks)
    """
    cryo = cryos[0]

    # Compute mask
    volume_mask, dilated_volume_mask= mask.masking_options(args.mask, means, volume_shape, dtype_real, args.mask_dilate_iter, args.keep_input_mask, args.dilated_mask_dilation_iters  )
    ## Always dump mean?
    ### Dump mean and mask to file
    output_folder = args.outdir + '/output/'
    o.mkdir_safe(output_folder)
    o.mkdir_safe(output_folder + 'volumes/')
    o.save_volume(means['combined'], output_folder + 'volumes/' + 'mean', volume_shape, from_ft = True,  voxel_size = cryos[0].voxel_size)
    o.save_volume(means['corrected0'], output_folder + 'volumes/' + 'mean_half1_unfil', volume_shape, from_ft = True,  voxel_size = cryos[0].voxel_size)
    o.save_volume(means['corrected1'], output_folder + 'volumes/' + 'mean_half2_unfil', volume_shape, from_ft = True,  voxel_size = cryos[0].voxel_size)
    o.save_volume(volume_mask, output_folder + 'volumes/' + 'mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)

    # Filter and save mean
    from recovar import locres
    half1 = ftu.get_idft3(means['corrected0'].reshape(cryos[0].volume_shape))
    half2 = ftu.get_idft3(means['corrected1'].reshape(cryos[0].volume_shape))
    best_filtered_nob, _, _, _, _ = locres.local_resolution(half1, half2, 0, cryos[0].voxel_size, use_filter = True, fsc_threshold = 1/7, use_v2 = True)
    o.save_volume(best_filtered_nob, output_folder + 'volumes/' + 'mean_filt', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)

    ### FIGURE OUT MASKS
    if args.focus_mask is not None:
        focus_mask, _= mask.masking_options(args.focus_mask, means, volume_shape, dtype_real, args.mask_dilate_iter, args.keep_input_mask)
    else:
        focus_mask = volume_mask

    if args.use_complement_mask:
        complement_mask = (volume_mask > 0.90)*1.0 - (focus_mask > 0.9)*1.0
        complement_mask = (complement_mask > 0)
        from recovar import mask as mask_fn
        complement_mask = np.array(mask_fn.soften_volume_mask(complement_mask, 3).astype(np.float32))
        focus_masks = [complement_mask, focus_mask]
    else:
        focus_masks = [focus_mask]

    return volume_mask, dilated_volume_mask, focus_masks


def stage_noise_refine_and_variance(cryo, cryos, means, batch_size, dilated_volume_mask, args, noise_model):
    """
    Noise estimation (multiple methods) and variance computation.

    Corresponds to pipeline.py lines 526-595.

    Returns:
        (noise_var_used, variance_est, variance_fsc, noise_p_variance_est,
         radial_noise_var_outside_mask, radial_ub_noise_var,
         white_noise_var_outside_mask, image_PS, std_image_PS,
         masked_image_PS, std_masked_image_PS, ub_noise_var_by_var_est)
    """
    ## NOW ESTIMATE NOISE A FEW WAYS

    noise_time = time.time()
    use_new_noise_fn = args.new_noise_est or args.premultiplied_ctf
    logger.info(f"Using new noise estimation function?: {use_new_noise_fn}")

    ## First, estimate noise outside the mask
    if args.mask.endswith(".mrc"):
        if use_new_noise_fn:
            masked_image_PS, image_PS = noise.fit_noise_model_to_images(cryo, dilated_volume_mask, means['combined'], None, batch_size=batch_size, invert_mask = True, disc_type = 'linear_interp')
        else:
            masked_image_PS, _,_ =  noise.estimate_noise_variance_from_outside_mask_v2(cryo, dilated_volume_mask, batch_size)
            white_noise_var_outside_mask = noise.estimate_white_noise_variance_from_mask(cryo, dilated_volume_mask, batch_size)
            _, _, image_PS, _ =  noise.estimate_radial_noise_statistic_from_outside_mask(cryo, dilated_volume_mask, batch_size)
        # white_noise_var_outside_mask = white_noise_var_outside_mask.copy()
        std_image_PS = None
        std_masked_image_PS = None
    else:
        if use_new_noise_fn:
            # radial_noise_var_outside_mask =
            masked_image_PS, image_PS = noise.fit_noise_model_to_images(cryo, dilated_volume_mask, means['combined'], None, batch_size=batch_size, invert_mask = True, disc_type = 'linear_interp')
            print("change discretization to cubic!")
            std_masked_image_PS = None
            std_image_PS = None

        else:
            masked_image_PS, std_masked_image_PS, image_PS, std_image_PS =  noise.estimate_radial_noise_statistic_from_outside_mask(cryo, dilated_volume_mask, batch_size)

    radial_noise_var_outside_mask = masked_image_PS
    white_noise_var_outside_mask = np.median(masked_image_PS)

    if use_new_noise_fn:
        assert (noise_model == "radial" or noise_model == "radial_per_tilt"), f"new noise fn only works with radial noise model. You set {noise_model}"


    logger.info(f"time to estimate noise is {time.time() - noise_time}")
    utils.report_memory_device(logger=logger)

    ## Then, estimate the EX [ (y_i - P_i \mu)^2] = noise_variance + signal_variance *CTF^2, where CTF is the CTF of the image, so this is the upper bound on the noise variance.
    noise_time = time.time()
    if use_new_noise_fn:
        # radial_noise_var_outside_mask =
        radial_ub_noise_var, _ =  noise.fit_noise_model_to_images(cryo, dilated_volume_mask, means['combined'], None, batch_size=batch_size, invert_mask = False, disc_type = 'linear_interp')
    else:
        radial_ub_noise_var, _,_ =  noise.estimate_radial_noise_upper_bound_from_inside_mask_v2(cryo, means['combined'], dilated_volume_mask, batch_size)

    logger.info(f"time to upper bound noise is {time.time() - noise_time}")

    ## Use this to bound the noise variance.
    utils.report_memory_device(logger=logger)
    radial_noise_var_ubed = np.where(radial_noise_var_outside_mask >  radial_ub_noise_var, radial_ub_noise_var, radial_noise_var_outside_mask)

    if noise_model == "white":
        noise_var_used = np.ones_like(radial_noise_var_ubed) * white_noise_var_outside_mask
    else:
        noise_var_used = radial_noise_var_ubed

    if (noise_var_used <0).any():
        logger.info("Negative noise variance detected. Setting to image power spectrum / 10")

    noise_var_used = np.where(noise_var_used < 0, image_PS / 10, noise_var_used)
    noise.update_noise_variance(noise_var_used, cryos)

    # This works in place?
    variance_est, ub_noise_var_by_var_est = noise.upper_bound_noise_by_signal_p_noise_dispatched(noise_var_used, cryos, means, batch_size, dilated_volume_mask)
    # This computes the variance once more which is unnecessary. TODO: rewrite this
    variance_est, _, variance_fsc, _, noise_p_variance_est = covariance_estimation.compute_variance(cryos, means['combined'], batch_size//2, dilated_volume_mask, use_regularization = True, disc_type = 'cubic')

    return (noise_var_used, variance_est, variance_fsc, noise_p_variance_est,
            radial_noise_var_outside_mask, radial_ub_noise_var,
            white_noise_var_outside_mask, image_PS, std_image_PS,
            masked_image_PS, std_masked_image_PS, ub_noise_var_by_var_est)


def stage_covariance_pca(cryos, options, means, mean_prior, focus_masks, dilated_volume_mask, valid_idx, batch_size, gpu_memory, variance_est, args):
    """
    Covariance options setup, optional test_covar_options, PCA loop over focus masks, cleanup.

    Corresponds to pipeline.py lines 642-743.

    Returns:
        (u, s, covariance_cols, picked_frequencies, column_fscs, covariance_options)
    """
    if args.test_covar_options:
        tests = [ ]
        idx = 0
        for test in tests:
            output_folder = args.outdir + '/output/'
            # Compute principal components
            covariance_options = covariance_estimation.get_default_covariance_computation_options(cryos[0].grid_size)
            for key in test:
                covariance_options[key] = test[key]

            u,s, covariance_cols, picked_frequencies, column_fscs = principal_components.estimate_principal_components(cryos, options, means, mean_prior, focus_mask, dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use=gpu_memory, covariance_options = covariance_options, variance_estimate = variance_est['combined'])
            from recovar import output
            output.mkdir_safe(output_folder)
            utils.pickle_dump({
                'options':test, 'u' :u['rescaled'][:,:20], 's' :s['rescaled'][:20], 'picked_frequencies':picked_frequencies
            }, output_folder + f'test_{idx}.pkl')
            del u, s, covariance_cols, picked_frequencies, column_fscs
            idx = idx + 1
            print('done with', idx, test)


    utils.report_memory_device(logger=logger)

    covariance_options = covariance_estimation.get_default_covariance_computation_options(cryos[0].grid_size)
    if args.low_memory_option:
        covariance_options['sampling_n_cols'] = 50
        covariance_options['randomized_sketch_size'] = 100
        covariance_options['n_pcs_to_compute'] = 100
        covariance_options['sampling_avoid_in_radius'] = 3

    if args.very_low_memory_option:
        covariance_options['sampling_n_cols'] = 25
        covariance_options['randomized_sketch_size'] = 35
        covariance_options['n_pcs_to_compute'] = 30
        covariance_options['sampling_avoid_in_radius'] = 3

    if args.dont_use_image_mask:
        covariance_options['mask_images_in_proj'] = False
        covariance_options['mask_images_in_H_B'] = False


    # Compute principal components
    # Only focus_mask[-1] will do a zdim search, rest will do zdim_for_rest

    # if len(focus_mask) > 1:
    num_foc_masks = len(focus_masks)
    u = []
    s = []
    # This could be sped up by a factor of len(focus_masks)
    zdim_for_rest = 20 # Maybe should make this an option
    n_pcs_to_keep = np.max( np.append(options['zs_dim_to_test'], 50))

    ## FIXME
    ignore_zero_frequency = options['ignore_zero_frequency']
    # mean_for_contrast_correction = means['combined_regularized'] if args.contrast_use_reg_mean else means['combined']

    # options['ignore_zero_frequency'] = False
    for idx, focus_mask in enumerate(focus_masks):
        u_this,s_this, covariance_cols, picked_frequencies, column_fscs = principal_components.estimate_principal_components(cryos, options, means, mean_prior, focus_mask, dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use=gpu_memory, covariance_options = covariance_options, variance_estimate = variance_est['combined'], use_reg_mean_in_contrast = args.use_reg_mean_in_contrast)
        if idx == num_foc_masks -1:
            s.append(s_this['rescaled'][:n_pcs_to_keep].copy())
            u.append(u_this['rescaled'][:,:n_pcs_to_keep].copy())
        else:
            s.append(s_this['rescaled'][:zdim_for_rest].copy())
            u.append(u_this['rescaled'][:,:zdim_for_rest].copy())
        del u_this, s_this
    u = { 'rescaled' : np.concatenate(u, axis = 1), 'real' : None}
    s =  { 'rescaled' : np.concatenate(s, axis = 0), 'real': None}
    options['ignore_zero_frequency'] = ignore_zero_frequency

    # Check if u and s are finite and not NaN
    if not np.all(np.isfinite(u['rescaled'])):
        raise ValueError("u contains non-finite values")
    if not np.all(np.isfinite(s['rescaled'])):
        raise ValueError("s contains non-finite values")

    # Check if s is positive
    if not np.all(s['rescaled'] > 0):
        raise ValueError("s contains non-positive values")

    # Check if u and s are of dtype float32/complex64
    if u['rescaled'].dtype not in [np.float32, np.complex64]:
        raise TypeError(f"u is not of dtype float32 or complex64, but {u['rescaled'].dtype}")
    if s['rescaled'].dtype not in [np.float32, np.complex64]:
        raise TypeError(f"s is not of dtype float32 or complex64, but {s['rescaled'].dtype}")

    # Check if mask is of dtype float32
    # (volume_mask check is done in the caller after this returns)

    if not args.keep_intermediate:
        del u['real']
        if 'rescaled_no_contrast' in u:
            del u['rescaled_no_contrast']
        covariance_cols = None

    return (u, s, covariance_cols, picked_frequencies, column_fscs, covariance_options)


def stage_embedding(cryos, means, u, s, volume_mask, gpu_memory, options, focus_masks, noise_var_used):
    """
    Compute per-image embeddings for each zdim.

    Corresponds to pipeline.py lines 732-766.

    Returns:
        (zs, cov_zs, est_contrasts)
    """
    num_foc_masks = len(focus_masks)
    zdim_for_rest = 20

    if options['ignore_zero_frequency']:
        # Make the noise in 0th frequency gigantic. Effectively, this ignore this frequency when fitting.
        logger.info('ignoring zero frequency')
        noise_var_used[0] *=1e16

    # image_cov_noise = np.asarray(noise.make_radial_noise(noise_var_used, cryos[0].image_shape))

    # Compute embeddings
    zs = {}; cov_zs = {}; est_contrasts = {}
    for zdim in options['zs_dim_to_test']:
        # Now we keep num_foc_masks-1*zdim_rest + zdim
        n_pcs_to_use = (num_foc_masks-1)*zdim_for_rest + zdim
        z_time = time.time()
        zs[zdim], cov_zs[zdim], est_contrasts[zdim], _ = embedding.get_per_image_embedding(means['combined'], u['rescaled'], s['rescaled'] , n_pcs_to_use,
                                                                 cryos, volume_mask, gpu_memory, 'linear_interp',
                                                                contrast_grid = None, contrast_option = options['contrast'],
                                                                ignore_zero_frequency = options['ignore_zero_frequency'] )
        logger.info(f"embedding time for zdim={zdim}: {time.time() - z_time}")


    for zdim in options['zs_dim_to_test']:
        z_time = time.time()
        n_pcs_to_use = (num_foc_masks-1)*zdim_for_rest + zdim
        key = f"{zdim}_noreg"
        zs[key], cov_zs[key], est_contrasts[key], _ = embedding.get_per_image_embedding(means['combined'], u['rescaled'], s['rescaled']* 0 + np.inf , n_pcs_to_use,
                                                                 cryos, volume_mask, gpu_memory, 'linear_interp',
                                                                contrast_grid = None, contrast_option = options['contrast'],
                                                                ignore_zero_frequency = options['ignore_zero_frequency'] )
        logger.info(f"embedding time for zdim={zdim}_noreg: {time.time() - z_time}")

    return zs, cov_zs, est_contrasts


def stage_save(args, cryos, means, u, s, volume_mask, dilated_volume_mask,
               focus_masks, zs, cov_zs, est_contrasts, noise_var_from_hf,
               noise_var_used, noise_var_from_het_residual,
               radial_noise_var_outside_mask, radial_ub_noise_var,
               white_noise_var_outside_mask, ub_noise_var_by_var_est,
               image_PS, std_image_PS, masked_image_PS, std_masked_image_PS,
               variance_est, variance_fsc, noise_p_variance_est,
               covariance_cols, covariance_options, column_fscs,
               picked_frequencies, contrasts_for_second, options, ind_split,
               path_mapping, st_time):
    """
    Post-loop: heterogeneity noise estimation, results assembly, saving, plotting.

    Corresponds to pipeline.py lines 773-954.
    """
    volume_shape = cryos[0].volume_shape
    batch_size = utils.get_image_batch_size(cryos[0].grid_size, utils.get_gpu_memory_total())

    num_foc_masks = len(focus_masks)
    zdim_for_rest = 20

    zs_cont = {}; cov_zs_cont = {}; est_contrasts_cont = {}
    var_metrics = {'filt_var': None}

    zdim = np.max(options['zs_dim_to_test'])

    if not args.tilt_series:
        n_pcs_to_use = (num_foc_masks-1)*zdim_for_rest + zdim
        noise_var_from_het_residual, _,_ = noise.estimate_noise_from_heterogeneity_residuals_inside_mask_v2(cryos[0], dilated_volume_mask, means['combined'], u['rescaled'][:,:n_pcs_to_use], est_contrasts[zdim], zs[zdim], batch_size//10, disc_type = covariance_options['disc_type'] )
    else:
        noise_var_from_het_residual = None
    # ### END OF DEL

    logger.info(f"embedding time: {time.time() - st_time}")

    utils.report_memory_device()

    # Dump results to file
    output_model_folder = args.outdir + '/model/'
    o.mkdir_safe(args.outdir)
    o.mkdir_safe(output_model_folder)

    logger.info(f"peak gpu memory use {utils.get_peak_gpu_memory_used(device =0)}")


    # For now, maybe just dump the rest?

    if args.use_complement_mask:
        import copy
        zs_full = copy.deepcopy(zs)
        for key in zs:
            zs[key] = zs[key][:,zdim_for_rest:]
        for key in cov_zs:
            cov_zs[key] = cov_zs[key][:,zdim_for_rest:,zdim_for_rest:]
        u['rescaled'] = u['rescaled'][:,zdim_for_rest:]
        s['rescaled'] = s['rescaled'][zdim_for_rest:]




    output_folder = args.outdir + '/output/'
    o.mkdir_safe(output_folder)
    o.save_covar_output_volumes(output_folder, means['combined'], u['rescaled'], s, volume_mask, volume_shape,  voxel_size = cryos[0].voxel_size)
    o.save_volume(volume_mask, output_folder + 'volumes/' + 'mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)
    o.save_volume(dilated_volume_mask, output_folder + 'volumes/' + 'dilated_mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)
    # Note: focus_mask here refers to focus_masks[-1] as in original code
    focus_mask = focus_masks[-1]
    o.save_volume(focus_mask, output_folder + 'volumes/' + 'focus_mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)
    if args.use_complement_mask:
        o.save_volume(focus_masks[0], output_folder + 'volumes/' + 'complement_mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)


    embedding_dict = { 'zs': zs, 'cov_zs' : cov_zs , 'contrasts': est_contrasts, 'zs_cont' : zs_cont, 'cov_zs_cont' : cov_zs_cont, 'contrasts_cont' : est_contrasts_cont}

    if args.tilt_series:
        particles_ind_split = [ cryo.image_stack.dataset_tilt_indices for cryo in cryos]
    else:
        particles_ind_split = ind_split

    utils.pickle_dump(particles_ind_split, output_model_folder + 'particles_halfsets.pkl')
    pickle.dump(ind_split, open(output_model_folder + 'halfsets.pkl', 'wb'))
    args.halfsets = output_model_folder + 'particles_halfsets.pkl'

    # Organize results into logical sections
    result = {
        # Core reconstruction results
        's': s['rescaled'],
        's_all': s,
        'density': None,
        'version': '0.5',

        # Volume metadata
        'volume_shape': volume_shape,
        'voxel_size': cryos[0].voxel_size,

        # Noise estimation results
        'noise_var_from_hf': noise_var_from_hf,
        'noise_var_from_het_residual': np.array(noise_var_from_het_residual),
        'noise_var_used': np.array(noise_var_used),
        'radial_noise_var_outside_mask': np.array(radial_noise_var_outside_mask),
        'radial_ub_noise_var': np.array(radial_ub_noise_var),
        'white_noise_var_outside_mask': np.array(white_noise_var_outside_mask),
        'ub_noise_var_by_var_est': np.array(ub_noise_var_by_var_est),

        # Power spectrum measurements
        'image_PS': np.array(image_PS),
        'std_image_PS': np.array(std_image_PS) if std_image_PS is not None else None,
        'masked_image_PS': np.array(masked_image_PS),
        'std_masked_image_PS': np.array(std_masked_image_PS) if std_masked_image_PS is not None else None,

        # Variance and covariance results
        'variance_est': variance_est,
        'variance_fsc': variance_fsc,
        'noise_p_variance_est': noise_p_variance_est,
        'covariance_cols': None,
        'covariance_options': covariance_options,

        # FSC and frequency analysis
        'column_fscs': column_fscs,
        'picked_frequencies': picked_frequencies,

        # PCA and metrics
        'pc_metric': var_metrics['filt_var'],
        'contrasts_for_second': contrasts_for_second,
        'latent_space_bounds': None,  # np.array(latent_space_bounds)

        # Input parameters
        'input_args': args
    }

    # Add original paths information if data was copied
    if path_mapping is not None:
        result['original_paths'] = path_mapping

        # Restore original paths in input_args before saving (only for paths that were copied)
        logger.info("Restoring original paths in input_args before saving...")

        # Only restore paths that were actually copied (have both original and temp entries)
        paths_to_restore = [
            ('original_particles', 'temp_particles', 'particles'),
            ('original_poses', 'temp_poses', 'poses'),
            ('original_ctf', 'temp_ctf', 'ctf'),
            ('original_mask', 'temp_mask', 'mask'),
            ('original_focus_mask', 'temp_focus_mask', 'focus_mask'),
            ('original_ind', 'temp_ind', 'ind'),
            ('original_tilt_ind', 'temp_particle_ind', 'tilt_ind'),
            ('original_halfsets', 'temp_halfsets', 'halfsets'),
        ]

        for orig_key, temp_key, attr_name in paths_to_restore:
            if orig_key in path_mapping and temp_key in path_mapping:
                # This path was copied, so restore the original
                setattr(args, attr_name, path_mapping[orig_key])
                logger.info(f"Restored {attr_name} path: {path_mapping[orig_key]}")
            elif orig_key in path_mapping:
                # Only original exists, meaning it wasn't copied (e.g., datadir)
                if attr_name == 'datadir' and path_mapping[orig_key]:
                    setattr(args, attr_name, path_mapping[orig_key])
                    logger.info(f"Restored {attr_name} path: {path_mapping[orig_key]}")
                else:
                    logger.debug(f"Skipping {attr_name} - not copied to temp location")

    # Convert all non-None values to numpy arrays where possible
    for key, value in result.items():
        if value is not None and not isinstance(value, (dict, str, float, int, bool)):
            try:
                result[key] = np.array(value)
            except (TypeError, ValueError) as e:
                # Log if conversion fails but continue
                print(f"Warning: Could not convert {key} to numpy array: {e}")

    utils.pickle_dump(result, output_model_folder + 'params.pkl')
    utils.pickle_dump(covariance_cols, output_model_folder + 'covariance_cols.pkl')


    for entry in embedding_dict:
        for key in embedding_dict[entry]:
            if entry == 'contrasts' and args.tilt_series and ('shared' not in options['contrast']):
                embedding_dict[entry][key] = dataset.reorder_to_original_indexing_from_halfsets(embedding_dict[entry][key], ind_split)
            else:
                embedding_dict[entry][key] = dataset.reorder_to_original_indexing_from_halfsets(embedding_dict[entry][key], particles_ind_split)

            # for k in range(num_foc_masks-1):
            #     embedding_dict[entry][key][k] = embedding_dict[entry][key][k].astype(np.float32)
    utils.pickle_dump(embedding_dict, output_model_folder + 'embeddings.pkl')
    if args.use_complement_mask:
        utils.pickle_dump(zs_full, output_model_folder + 'zs_with_complement.pkl')

    logger.info(f"Dumped results to file:, {output_model_folder}results.pkl")
    logger.info(f"total time: {time.time() - st_time}")

    # Clean up temp files at the end
    if path_mapping is not None and not args.no_cleanup:
        cleanup_temp_files(path_mapping)


    from recovar import output
    po = output.PipelineOutput(args.outdir + '/')
    zdims = np.array(args.zdim)
    zdim_choose = np.argmin(np.abs(zdims - 10))
    zdim = zdims[zdim_choose]
    output.standard_pipeline_plots(po, zdim, args.outdir + '/output/plots/')
