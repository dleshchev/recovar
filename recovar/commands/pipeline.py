# If you want to extend and use recovar, you should import this first
import logging
# It is important to import cryodrgn before the setting basicConfig which is why it is imported here (but not used)
# import cryodrgn
logger = logging.getLogger(__name__)
import recovar.config 
import jax
import jax.numpy as jnp
import numpy as np
import os, argparse, time, pickle, sys, shutil
from recovar import output as o
from recovar import dataset, homogeneous, embedding, principal_components, latent_density, mask, utils, constants, noise, output, covariance_estimation
from recovar.fourier_transform_utils import fourier_transform_utils
from recovar.utils_core import copy_data_to_temp_folder, save_original_paths_info, cleanup_temp_files
ftu = fourier_transform_utils(jnp)
# logger.setLevel(logger.info)
logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "particles",
        type=os.path.abspath,
        help="Input particles (.mrcs, .star, .cs, or .txt)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Output directory to save model",
    )

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser.add_argument('--zdim', type=list_of_ints, default=[1,2,4,10,20], help="Dimensions of latent variable. Default=1,2,4,10,20")

    # parser.add_argument(
    #     "--zdim", type=list, help="Dimension of latent variable"
    # )
    parser.add_argument(
        "--poses", type=os.path.abspath, required=True, help="Image poses (.pkl)"
    )
    parser.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, required=True, help="CTF parameters (.pkl)"
    )

    # parser.add_argument(
    #     "--mask", metavar="mrc", default=None, type=os.path.abspath, help="mask (.mrc)"
    # )

    parser.add_argument(
        "--mask", metavar="mrc", required=True, help="solvent mask (.mrc).Can solve provide: from_halfmaps, sphere, none" 
    )

    parser.add_argument(
        "--focus-mask", metavar="mrc", dest = "focus_mask", default=None, type=os.path.abspath, help="focus mask (.mrc)"
    )

    parser.add_argument(
        "--keep-input-mask", action="store_true", dest="keep_input_mask", help="By default, the software thresholds and then softens mask. If this option is on, the input mask is used as is." 
    )

    parser.add_argument(
        "--use-complement-mask", action="store_true", dest = "use_complement_mask", help="Use complement of focus mask"
    )

    parser.add_argument(
        "--copy-to-folder", dest="copy_to_folder", default = None, type=os.path.abspath, help="Copy all input data files to this temporary folder before processing. Original paths will be saved in output."
    )

    # parser.add_argument(
    #     "--mask-option", metavar=str, default="input", help="mask options: from_halfmaps , input (default), sphere, none"
    # )

    parser.add_argument(
        "--mask-dilate-iter", type=int, default=0, dest="mask_dilate_iter", help="mask options how many iters to dilate solvent and focus mask"
    )


    parser.add_argument(
        "--correct-contrast",
        dest = "correct_contrast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="estimate and correct for amplitude scaling (contrast) variation across images. Default = false "
    )

    parser.add_argument(
        "--ignore-zero-frequency",
        dest = "ignore_zero_frequency",
        action="store_true",
        help="use if you want zero frequency to be ignored. If images have been normalized to 0 mean, this is probably a good idea"
    )

    # parser.add_argument(
    #     "--no-z-regularization",
    #     dest = "no_z_regularization",
    #     action="store_true",
    # )

    group = parser.add_argument_group("Dataset loading")
    group.add_argument(
        "--ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter images by these indices",
    )

    group.add_argument(
        "--particle-ind",
        dest="tilt_ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter particles by these indices (only for tilt-series/cryo-ET)",
    )


    group.add_argument(
        "--uninvert-data",
        dest="uninvert_data",
        default = "automatic",
        help="Invert data sign: options: true, false, automatic (default). automatic will swap signs if sum(estimated mean) < 0",
    )

    # group.add_argument(
    #     "--rerescale",
    #     dest = "rerescale",
    #     action="store_true",
    # )

    # Should probably add these options
    # group.add_argument(
    #     "--no-window",
    #     dest="window",
    #     action="store_false",
    #     help="Turn off real space windowing of dataset",
    # )
    # group.add_argument(
    #     "--window-r",
    #     type=float,
    #     default=0.85,
    #     help="Windowing radius (default: %(default)s)",
    # )
    group.add_argument(
        "--lazy",
        action="store_true",
        help="Lazy loading if full dataset is too large to fit in memory",
    )

    group.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file. If not specified, uses the directory of the star file.",
    )
    
    parser.add_argument(
        "--strip-prefix",
        help="Path prefix to strip from filenames in star file (used in starfile input ONLY). \
        Useful when star file contains longer paths than available on the system. By default, it strips the full path (except the filename). E.g, if you starfile path is Extract/job193/Subtomograms/XXX/XXX.mrcs, \
        and your directory looks like /your/path/to/Subtomograms, then you can use --strip-prefix Extract/job193 --datadir /your/path/to/.",
    )
    
    group.add_argument(
            "--n-images",
            default = -1,
            dest="n_images",
            type=int,
            help="Number of images to use (should only use for quick run)",
        )
    
    group.add_argument(
            "--padding",
            type=int,
            default = 0,
            help="Real-space padding",
        )
    
    group.add_argument(
            "--halfsets",
            default = None,
            type=os.path.abspath,
            help="Path to a file with indices of split dataset (.pkl).",
        )

    ### CHANGE THESE TWO BACK!?!?!?!
    group.add_argument(
            "--keep-intermediate",
            dest = "keep_intermediate",
            action="store_true",
            help="saves some intermediate result. Probably only useful for debugging"
        )

    group.add_argument(
            "--noise-model",
            dest = "noise_model",
            default = "radial",
            help="what noise model to use. Options are radial (default) computed from outside the masks, and white computed by power spectrum at high frequencies"
        )

    group.add_argument(
            "--mean-fn",
            dest = "mean_fn",
            default = "triangular",
            help="which mean function to use. Options are triangular (default), old, triangular_reg"
        )
    
    group.add_argument(
        "--accept-cpu",
        dest="accept_cpu",
        action="store_true",
        help="Accept running on CPU if no GPU is found",
    )

    group.add_argument(
            "--test-covar-options",
            dest = "test_covar_options",
            action="store_true",
            help="Only for development. Test different covariance estimation options"
        )

    group.add_argument(
            "--low-memory-option",
            help = "Use lower memory options for covariance estimation",
            dest = "low_memory_option",
            action="store_true",
        )


    group.add_argument(
            "--very-low-memory-option",
            help = "Use lowest memory options for covariance estimation",
            dest = "very_low_memory_option",
            action="store_true",
        )

    group.add_argument(
            "--dont-use-image-mask",
            dest = "dont_use_image_mask",
            action="store_true",
        )

    # group.add_argument(
    #         "--do-over-with-contrast",
    #         dest = "do_over_with_contrast",
    #         default = "True",
    #         help="Whether to run again once constrast is estimated",
    #     )
    
    parser.add_argument(
        "--tilt-series", action="store_true",  dest="tilt_series", help="Whether to use tilt_series."
    )

    parser.add_argument(
        "--tilt-series-ctf", default = None,  dest="tilt_series_ctf", help="What CTF to use for tilt series. Options : cryoem, relion5, warp (windows). Warptools is not yet supported. Default = cryoem if tilt series is False, relion5 if tilt series is True"
    )

    parser.add_argument(
        "--dose-per-tilt", default =None, type = float, dest="dose_per_tilt", help="Default = None, read from starfile"
    )

    parser.add_argument(
        "--angle-per-tilt", default =None,  type = float, dest="angle_per_tilt", help="Default = None, estimated from starfile"
    )

    # parser.add_argument(
    #     "--per-image-bfac-scale", action="store_true",
    # )

    parser.add_argument(
        "--only-mean", action="store_true", dest = "only_mean", help="Only compute mean"
    )

    parser.add_argument(
        "--ntilts", default = None, type=int, help="Number of tilts to use per tilt series. None = all (default)"
    )

    parser.add_argument(
        "--gpu-gb", default =None,  type = float, dest="gpu_memory", help="How much GPU memory to use. Default = all" 
    )

    parser.add_argument(
        "--premultiplied-ctf", dest = 'premultiplied_ctf', action="store_true", help="Whether to use premultiplied CTF. Default = False"
    )

    parser.add_argument(
        "--new-noise-est", dest = 'new_noise_est', action="store_true", help="Whether to use new noise estimation. Default = False"
    )


    parser.add_argument('--shared_contrast_across_tilts', action=argparse.BooleanOptionalAction, default =False,
                        help="Whether to share contrast (amplitude scale) across tilts in cryoET. Default = False")

    parser.add_argument('--use_reg_mean_in_contrast', action=argparse.BooleanOptionalAction, default =True)

    parser.add_argument(
            "--do-over-with-contrast",
            dest = "do_over_with_contrast",
            action=argparse.BooleanOptionalAction,
            default = None,
            help="Whether to run again once constrast is estimated. By default == correct_contrast. Can enter --no-do-over-with-contrast to turn off",
        )
    

    parser.add_argument('--dilated-mask-dilation-iters', 
                        type = int,
                        default = None,
                        help = "How many times to dilate the mask. Default = 6 * volume_shape[0] / 128"
                        )
                    

    parser.add_argument("--no-cleanup", action="store_true", help="Do not clean up temporary files after processing (useful for chaining multiple pipeline calls)")

    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default=None,
        type=os.path.abspath,
        help="Checkpoint directory for stage results, enabling resume. "
             "Default: {outdir}/checkpoint. "
             "Override with RECOVAR_CHECKPOINT_DIR env var."
    )
    parser.add_argument(
        "--resume-from-stage",
        dest="resume_from_stage",
        default=None,
        type=int,
        help="Resume from stage N (skip stages 0..N-1, requires their checkpoints)."
    )
    parser.add_argument(
        "--keep-checkpoints",
        dest="keep_checkpoints",
        action="store_true",
        default=False,
        help="Keep checkpoint directories after successful completion."
    )

    return parser
    

def _resolve_checkpoint_dir(args):
    """Determine checkpoint directory from CLI > env var > default."""
    if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir is not None:
        return args.checkpoint_dir
    env_dir = os.environ.get("RECOVAR_CHECKPOINT_DIR")
    if env_dir is not None:
        return os.path.abspath(env_dir)
    return os.path.join(args.outdir, "checkpoint")


def standard_recovar_pipeline(args):
    from recovar import stages
    from recovar.stage_checkpoint import StageCheckpoint

    st_time = time.time()

    # Resolve checkpoint directory
    checkpoint_dir = _resolve_checkpoint_dir(args)
    use_checkpoints = (hasattr(args, 'checkpoint_dir') and args.checkpoint_dir is not None) or \
                      os.environ.get("RECOVAR_CHECKPOINT_DIR") is not None or \
                      (hasattr(args, 'resume_from_stage') and args.resume_from_stage is not None)

    if use_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpointing enabled: {checkpoint_dir}")

    def ckpt(name):
        """Create a StageCheckpoint if checkpointing is enabled, else return None."""
        if use_checkpoints:
            return StageCheckpoint(checkpoint_dir, name)
        return None

    def stage_complete(c):
        """Check if a stage checkpoint exists and is complete."""
        return c is not None and c.is_complete()

    # --- Stage 0: Setup ---
    c_setup = ckpt("stage_00_setup")
    if stage_complete(c_setup):
        logger.info("Skipping setup (checkpoint exists)")
        setup = c_setup.load_object("setup_result")
        cryos = dataset.get_split_datasets_from_dict(
            setup['dataset_loader_dict'], setup['ind_split'], args.lazy)
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
        # Re-initialize noise model
        for cryo in cryos:
            if noise_model == "radial":
                cryo.set_radial_noise_model(None)
            elif noise_model in ('radial_per_tilt', 'radial-per-tilt'):
                cryo.set_variable_radial_noise_model(None)
    else:
        (cryos, ind_split, options, batch_size, gpu_memory, noise_var_from_hf,
         valid_idx, noise_model, n_repeats, path_mapping, dataset_loader_dict) = stages.stage_setup(args)
        if c_setup is not None:
            c_setup.save_object("setup_result", {
                'ind_split': ind_split, 'options': options,
                'batch_size': batch_size, 'gpu_memory': gpu_memory,
                'noise_var_from_hf': noise_var_from_hf,
                'valid_idx': valid_idx, 'noise_model': noise_model,
                'n_repeats': n_repeats, 'path_mapping': path_mapping,
                'dataset_loader_dict': dataset_loader_dict,
            })
            c_setup.mark_complete()

    contrasts_for_second = None
    for repeat in range(n_repeats):
        ps = f"_pass{repeat}" if n_repeats > 1 else ""

        if repeat == 1:
            contrasts_for_second = stages.apply_contrast_correction(
                cryos, options, est_contrasts)
        else:
            contrasts_for_second = None

        # --- Stage 2: Mean ---
        c_mean = ckpt(f"stage_02_mean{ps}")
        if stage_complete(c_mean):
            logger.info("Skipping mean (checkpoint exists)")
            means = c_mean.load_object("means")
            mean_prior = c_mean.load_array("mean_prior")
            uninvert_applied = c_mean.load_config().get("uninvert_applied", False)
            if uninvert_applied:
                for cryo in cryos:
                    cryo.image_stack.mult = -1 * cryo.image_stack.mult
        else:
            means, mean_prior, uninvert_applied = stages.stage_mean(
                cryos, batch_size, noise_var_from_hf, args, st_time=st_time)
            if c_mean is not None:
                c_mean.save_object("means", means)
                c_mean.save_array("mean_prior", mean_prior)
                c_mean.save_config({"uninvert_applied": uninvert_applied})
                c_mean.mark_complete()

        # --- Stage 3: Mask ---
        c_mask = ckpt(f"stage_03_mask{ps}")
        if stage_complete(c_mask):
            logger.info("Skipping mask (checkpoint exists)")
            volume_mask, dilated_volume_mask, focus_masks = c_mask.load_object("mask_result")
        else:
            volume_mask, dilated_volume_mask, focus_masks = stages.stage_mask(
                args, means, cryos[0].volume_shape, cryos[0].dtype_real, cryos)
            if c_mask is not None:
                c_mask.save_object("mask_result", (volume_mask, dilated_volume_mask, focus_masks))
                c_mask.mark_complete()

        if args.only_mean:
            return

        if volume_mask.dtype != np.float32:
            raise TypeError(f"volume_mask is not of dtype float32, but {volume_mask.dtype}")

        # --- Stage 4: Noise + Variance ---
        c_noise = ckpt(f"stage_04_noise{ps}")
        if stage_complete(c_noise):
            logger.info("Skipping noise/variance (checkpoint exists)")
            (noise_var_used, variance_est, variance_fsc, noise_p_variance_est,
             radial_noise_var_outside_mask, radial_ub_noise_var,
             white_noise_var_outside_mask, image_PS, std_image_PS,
             masked_image_PS, std_masked_image_PS,
             ub_noise_var_by_var_est) = c_noise.load_object("noise_result")
            noise.update_noise_variance(noise_var_used, cryos)
        else:
            (noise_var_used, variance_est, variance_fsc, noise_p_variance_est,
             radial_noise_var_outside_mask, radial_ub_noise_var,
             white_noise_var_outside_mask, image_PS, std_image_PS,
             masked_image_PS, std_masked_image_PS,
             ub_noise_var_by_var_est) = stages.stage_noise_refine_and_variance(
                cryos[0], cryos, means, batch_size, dilated_volume_mask, args, noise_model)
            if c_noise is not None:
                c_noise.save_object("noise_result",
                    (noise_var_used, variance_est, variance_fsc, noise_p_variance_est,
                     radial_noise_var_outside_mask, radial_ub_noise_var,
                     white_noise_var_outside_mask, image_PS, std_image_PS,
                     masked_image_PS, std_masked_image_PS,
                     ub_noise_var_by_var_est))
                c_noise.mark_complete()

        # --- Stage 5+6: Covariance + PCA ---
        c_pca = ckpt(f"stage_06_pca{ps}")
        if stage_complete(c_pca):
            logger.info("Skipping covariance/PCA (checkpoint exists)")
            (u, s, covariance_cols, picked_frequencies, column_fscs,
             covariance_options) = c_pca.load_object("pca_result")
        else:
            (u, s, covariance_cols, picked_frequencies, column_fscs,
             covariance_options) = stages.stage_covariance_pca(
                cryos, options, means, mean_prior, focus_masks,
                dilated_volume_mask, valid_idx, batch_size, gpu_memory,
                variance_est, args)
            if c_pca is not None:
                c_pca.save_object("pca_result",
                    (u, s, covariance_cols, picked_frequencies, column_fscs,
                     covariance_options))
                c_pca.mark_complete()

        # --- Stage 7: Embedding ---
        c_embed = ckpt(f"stage_07_embedding{ps}")
        if stage_complete(c_embed):
            logger.info("Skipping embedding (checkpoint exists)")
            zs, cov_zs, est_contrasts = c_embed.load_object("embedding_result")
        else:
            zs, cov_zs, est_contrasts = stages.stage_embedding(
                cryos, means, u, s, volume_mask, gpu_memory, options,
                focus_masks, noise_var_used)
            if c_embed is not None:
                c_embed.save_object("embedding_result", (zs, cov_zs, est_contrasts))
                c_embed.mark_complete()

        if repeat == 1:
            for key in est_contrasts:
                est_contrasts[key] = est_contrasts[key] * contrasts_for_second

    # --- Stage 8: Save ---
    stages.stage_save(args, cryos, means, u, s, volume_mask,
                      dilated_volume_mask, focus_masks, zs, cov_zs,
                      est_contrasts, noise_var_from_hf, noise_var_used,
                      None,
                      radial_noise_var_outside_mask, radial_ub_noise_var,
                      white_noise_var_outside_mask, ub_noise_var_by_var_est,
                      image_PS, std_image_PS, masked_image_PS,
                      std_masked_image_PS, variance_est, variance_fsc,
                      noise_p_variance_est, covariance_cols, covariance_options,
                      column_fscs, picked_frequencies, contrasts_for_second,
                      options, ind_split, path_mapping, st_time)

    # Cleanup checkpoints unless asked to keep
    if use_checkpoints and not getattr(args, 'keep_checkpoints', False):
        import shutil
        logger.info(f"Cleaning up checkpoint directory: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    return means, u, s, volume_mask, dilated_volume_mask, noise_var_used




def main():
    # import jax
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    standard_recovar_pipeline(args)



if __name__ == "__main__":
    main()