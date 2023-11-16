import os
from os.path import join as jn
import sys
import subprocess
from subprocess import Popen
import argparse
from shutil import copyfile
from data_generation.temporal_causal3dident.npzify import npzify
from data_generation_causal3dident import generate_latents
from myutil import Namespace

from constants import P2_ROOT, CITRIS_ROOT, TC3DI_ROOT, BLENDER_PATH, IMG_GEN_SCRIPT_PATH

def add_deconf_ood_shared_args(parser):
    parser.add_argument('--n_points', type=int, default=100, help='Number of points')
    parser.add_argument('--max_valtest_points', type=int, default=10000, help='Max number of points for val and test')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--int_probs', type=float, default=0.1, help='Probability of intervention')  # 0.1 follows CITRIS paper
    parser.add_argument('--skip_latent_creation', action='store_true', help='Skip latent creation')
    parser.add_argument('--skip_image_generation', action='store_true', help='Skip image generation')
    parser.add_argument('--skip_npzification', action='store_true', help='Skip npzification')
    parser.add_argument('--skip_idx_permutation_creation', action='store_true', help='Skip idx permutation creation')
    parser.add_argument('--exclude_objects', type=int, nargs='+', default=None,
                        help='List of objects to exclude for generating the dataset, e.g. to test the generalization to unseen shapes.')
    parser.add_argument('--batch_size', type=int, default=2500, help='Batch size')
    parser.add_argument('--overwrite_imgs', action='store_true', help='Overwrite images')
    # list of splits to generate
    parser.add_argument('--splits', type=str, nargs='+', default=["train", "val", "test"],
                        help='List of splits to generate')
    return parser
def add_deconf_specific_args(parser):
    parser.add_argument('--out_dir', type=str, default=TC3DI_ROOT, help='Output directory')
    return parser



def generate(args):
    # splits = ["val", "test"]
    # splits = ["train", "val", "test"]
    splits = args.splits
    print(splits)
    # splits = ["train"]

    ROOT_OUT_FOLDER = args.out_dir
    IID_PAIRS_FOLDER = jn(args.out_dir, "iid_pairs")
    maybe_no_intv = "_no_intv" if args.int_probs == 0.0 else ""
    DETAILED_OUTPUT_FOLDER = jn(IID_PAIRS_FOLDER, f"{args.n_points}_points{maybe_no_intv}")
    print(f"Output folder: {DETAILED_OUTPUT_FOLDER}")
    for split in splits:
        generate_split(args, split, DETAILED_OUTPUT_FOLDER, ROOT_OUT_FOLDER)


def generate_split(args, split, DETAILED_OUTPUT_FOLDER, ROOT_OUT_FOLDER):
    if args.max_valtest_points > 0 and split in ["val", "test"]:
        args.n_points = min(args.n_points, args.max_valtest_points)
        print(f"Setting n_points to {args.n_points} for {split}")
    if not args.skip_latent_creation:
        latent_generation_args = Namespace(
            n_points=args.n_points,
            output_folder=DETAILED_OUTPUT_FOLDER,
            coarse_vars=True,
            num_shapes=7,
            split=split,
            exclude_objects=args.exclude_objects,
            ood=False,
            iid_pairs=True,
            int_probs= args.int_probs,
        )
        generate_latents(latent_generation_args)
    else:
        print("Skipping latent creation")
    if not args.skip_image_generation:
        # BLENDER_PATH = jn(P2_ROOT, "blender-2.93.8-linux-x64", "blender")
        # IMG_GEN_SCRIPT_PATH = jn(CITRIS_ROOT, "data_generation", "temporal_causal3dident",
        #                           "generate_causal3dident_images.py")

        n_batches = args.n_points // args.batch_size if args.n_points > args.batch_size else 1
        print(f"n_batches: {n_batches}")

        processes = [Popen(
            f"{BLENDER_PATH} --background --python {IMG_GEN_SCRIPT_PATH} -- --output-folder {DETAILED_OUTPUT_FOLDER} --n-batches {n_batches} --batch-index {batch_index} --split {split} --iid_pairs" + (" --overwrite" if args.overwrite_imgs else ""),
            shell=True, stderr=subprocess.PIPE) for batch_index in range(n_batches)]
        for p in processes:
            p.wait()
    else:
        print("Skipping image generation")

    if not args.skip_npzification:
        npzify_args = Namespace(dir=DETAILED_OUTPUT_FOLDER, split=split, skip_imgs=args.skip_image_generation, iid_pairs=True)
        npzify(npzify_args)

        MINI_NUM = 100
        FULL_NUM = 10000 if (split != 'train') else 245000
        for NUM, maybe_mini in ((MINI_NUM, "_mini"), (FULL_NUM, "")):
            if args.n_points == NUM:
                if args.int_probs == 0.0:
                    copyfile(jn(DETAILED_OUTPUT_FOLDER, f"{split}.npz"), jn(ROOT_OUT_FOLDER, f"{split}_cma{maybe_mini}.npz"))
                    open(jn(ROOT_OUT_FOLDER, f"FYI_{split}_cma{maybe_mini}_comes_from_{NUM}_points_no_intv_in_iid_pairs.txt"), 'a').close()
                else:
                    copyfile(jn(DETAILED_OUTPUT_FOLDER, f"{split}.npz"), jn(ROOT_OUT_FOLDER, f"{split}_deconf{maybe_mini}.npz"))
                    open(jn(ROOT_OUT_FOLDER, f"FYI_{split}_deconf{maybe_mini}_comes_from_{NUM}_points_in_iid_pairs.txt"), 'a').close()
    else:
        print("Skipping npzification")


if __name__ == "__main__":
    # region argparse
    parser = argparse.ArgumentParser()
    # Add the arguments
    parser = add_deconf_ood_shared_args(parser)
    parser = add_deconf_specific_args(parser)

    # Parse the arguments
    args = parser.parse_args()
    # endregion
    generate(args)