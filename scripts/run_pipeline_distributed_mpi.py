"""Wrapper to run `recovar pipeline_distributed` under mpirun with the same
test-dataset conventions as scripts/run_test_distributed.sh, but driven by
env vars (RECOVAR_IMG_SIZE, RECOVAR_N_IMAGES, RECOVAR_LAZY) so the launcher
in submit_job.sh can pass them through `--env-file`-style.

Used by the dist-small-mpi-* / dist-large-mpi-* actions in submit_job.sh.
"""

import os
import sys
import traceback


def main():
    img_size = int(os.environ["RECOVAR_IMG_SIZE"])
    n_images = int(os.environ["RECOVAR_N_IMAGES"])
    lazy = os.environ.get("RECOVAR_LAZY", "").lower() in ("1", "true", "yes")

    dataset_dir = f"/workspace/data-{img_size}-{n_images}/test_dataset"
    os.chdir(dataset_dir)

    # Namespace the output directory by world size so successive
    # `dist-small-mpi-{1,2,4}node` runs don't clobber each other (they all
    # share the same dataset dir). SLURM_JOB_NUM_NODES is not propagated
    # through `mpirun -x` to remote ranks, so use MPI's own size.
    from mpi4py import MPI
    n_nodes = MPI.COMM_WORLD.Get_size()
    output_name = f"pipeline_distributed_mpi_output_{n_nodes}node"
    if lazy:
        output_name += "_lazy"

    sys.argv = [
        "recovar",
        f"particles.{img_size}.mrcs",
        "--ctf", "ctf.pkl",
        "--poses", "poses.pkl",
        "--mask=from_halfmaps",
        "-o", output_name,
    ]
    if lazy:
        sys.argv.append("--lazy")

    from recovar.commands.pipeline_distributed import main as recovar_main
    recovar_main()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        try:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        except Exception:
            sys.exit(1)
