import argparse
from os.path import join as jn
from subprocess import Popen
import subprocess
from constants import TC3DI_OOD_COMMON_ROOT, BLENDER_PATH, IMG_GEN_SCRIPT_PATH
from create_ood_idx_permutations_for_fsl import create_idx_permutations
from data_generation.temporal_causal3dident.data_generation_causal3dident import generate_latents
from data_generation.temporal_causal3dident.npzify import npzify
from generate_iid_pairs import add_deconf_ood_shared_args
from myutil import Namespace
import os


def add_ood_specific_args(parser):
    parser.add_argument('--out_dir', type=str, default=TC3DI_OOD_COMMON_ROOT, help='Output directory')
    parser.add_argument('--changed_mechs', type=str, nargs='+',
                        default=["x_pos", "y_pos", "z_pos", "alpha", "beta", "rot_spotlight", "hue_object",
                                 "hue_spotlight", "hue_background"], help='List of changed mechanisms')
    parser.add_argument('--do_at_once', action='store_true', help='Do everything at once')
    parser.add_argument('--iid_pairs', action='store_true', help='Generate iid pairs')
    return parser


def generate_ood(args):
    if args.do_at_once:
        print("Changing multiple mechanisms in one dataset")
        generate(args.changed_mechs, args)
    else:
        print("Changing one mechanism in multiple datasets")
        for changed_mech in args.changed_mechs:
            print(f"Changed mechanism: {changed_mech}")
            generate([changed_mech], args)


def generate(changed_mech_list, args):
    mechs_string = "__".join(changed_mech_list)
    maybe_no_intv = "_no_intv" if args.int_probs == 0.0 else ""
    n_points_subfolder = jn(args.out_dir, "iid_pairs" if args.iid_pairs else "",f"{args.n_points}_points{maybe_no_intv}")
    detailed_output_folder = jn(n_points_subfolder, mechs_string)
    # make dir if it doesn't exist
    if not os.path.exists(detailed_output_folder):
        os.makedirs(detailed_output_folder)
    print(f"Output folder: {detailed_output_folder}")
    for split in args.splits:
        print(f"Split: {split}")
        # If split is test or val, cap N_POINTS at n_points_capped=min(N_POINTS, 10000)
        if split == "train":
            n_points_capped = args.n_points
        else:
            n_points_capped = 10000
            if args.n_points < n_points_capped:
                n_points_capped = args.n_points
        print(f"n_points_capped: {n_points_capped}")
        # if local_changed_mechs starts with "shape", then OOD is empty, otherwise OOD is "--ood ${local_changed_mechs}"
        if changed_mech_list[0].startswith("shape"):
            ood_arg = []
        else:
            ood_arg = changed_mech_list
        if not args.skip_latent_creation:
            latent_generation_args = Namespace(
                n_points=n_points_capped,
                output_folder=detailed_output_folder,
                coarse_vars=True,
                num_shapes=7,
                split=split,
                exclude_objects=args.exclude_objects,
                ood=ood_arg if ood_arg != ['none'] else None,
                int_probs=args.int_probs,
                iid_pairs=args.iid_pairs
            )
            generate_latents(latent_generation_args)
        else:
            print("Skipping latent creation")
        if not args.skip_image_generation:
            n_batches = args.n_points // args.batch_size if args.n_points > args.batch_size else 1
            print(f"n_batches: {n_batches}")

            processes = [Popen(f"{BLENDER_PATH} --background --python {IMG_GEN_SCRIPT_PATH} -- --output-folder {detailed_output_folder} --n-batches {n_batches} --batch-index {batch_index} {'--iid_pairs' if args.iid_pairs else ''} --split {split}{' --overwrite' if args.overwrite_imgs else ''}", shell=True, stderr=subprocess.PIPE) for batch_index in range(n_batches)]
            for p in processes:
                p.wait()
        else:
            print("Skipping image generation")
        if not args.skip_npzification:
            npzify_args = Namespace(dir=detailed_output_folder, split=split, skip_imgs=args.skip_image_generation, iid_pairs=args.iid_pairs)
            npzify(npzify_args)
        else:
            print("Skipping npzification")

        if (not args.skip_idx_permutation_creation) and split in ["train", "val"]:
            idx_permutation_creation_args = Namespace(ood_data_dirs=[n_points_subfolder], split=split, n_points=n_points_capped)
            create_idx_permutations(idx_permutation_creation_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_deconf_ood_shared_args(parser)
    parser = add_ood_specific_args(parser)
    args = parser.parse_args()
    generate_ood(args)
