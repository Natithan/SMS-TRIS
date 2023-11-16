import argparse
from os.path import join as j
import numpy as np
import imageio
import glob
from myutil import namespace_to_arglist, Namespace
import multiprocessing as mp


def npzify(args: Namespace=None):

    args = parse_args(args)
    maybe_out = 'out_' if args.iid_pairs else ''
    maybe_full = 'full_' if args.triplet else ''
    ood_latents = np.load(j(args.dir, f'{maybe_full}{maybe_out}latents_{args.split}.npy'))
    ood_shape_latents = np.load(j(args.dir, f'{maybe_full}{maybe_out}shape_latents_{args.split}.npy'))
    ood_interventions = np.load(j(args.dir, f'interventions_{args.split}.npy'))
    if not args.skip_imgs:
        # ood_imgs = np.stack([imageio.v2.imread(impath) for impath in glob.glob(j(args.dir, f'{maybe_out}images_{args.split}', '*.png'))]) # this doesn't preserve order!!!
        # ood_imgs = np.stack([imageio.v2.imread(impath) for impath in sorted(glob.glob(j(args.dir, f'{maybe_out}images_{args.split}', '*.png')))]) # this preserves order, but is still slow
        max_workers = mp.cpu_count()
        with mp.Pool(max_workers) as pool:
            ood_imgs = np.stack(pool.map(imageio.v2.imread, sorted(glob.glob(j(args.dir, f'{maybe_out}images_{args.split}', '*.png')))))
    if args.iid_pairs:
        in_latents = np.load(j(args.dir, f'in_latents_{args.split}.npy'))
        in_shape_latents = np.load(j(args.dir, f'in_shape_latents_{args.split}.npy'))
        if not args.skip_imgs:
            # in_imgs = np.stack([imageio.v2.imread(impath) for impath in glob.glob(j(args.dir, f'in_images_{args.split}', '*.png'))]) # this doesn't preserve order!!!
            # in_imgs = np.stack([imageio.v2.imread(impath) for impath in sorted(glob.glob(j(args.dir, f'in_images_{args.split}', '*.png')))]) # this preserves order, but is still slow
            with mp.Pool(max_workers) as pool:
                in_imgs = np.stack(pool.map(imageio.v2.imread, sorted(glob.glob(j(args.dir, f'in_images_{args.split}', '*.png')))))
    if args.triplet:
        prev_imgs = np.load(j(args.dir, f'prev_images_{args.split}.npy'))
        ood_imgs = np.concatenate([prev_imgs, ood_imgs[:,None]], axis=1)
    npz_path = j(args.dir, f'{args.split}.npz')

    if not args.iid_pairs:
        np.savez(npz_path, interventions=ood_interventions, latents=ood_latents, shape_latents=ood_shape_latents,
             raw_latents=ood_latents, # These seem to be exactly the same as latents in the original data
             imgs=ood_imgs if not args.skip_imgs else None)
    else:
        np.savez(npz_path, interventions=ood_interventions, out_latents=ood_latents, out_shape_latents=ood_shape_latents,
                in_latents=in_latents, in_shape_latents=in_shape_latents,
                 in_imgs=in_imgs if not args.skip_imgs else None,
                 out_imgs=ood_imgs if not args.skip_imgs else None,
                 raw_out_latents=ood_latents, raw_in_latents=in_latents)


def parse_args(args: Namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--split', type=str,
                        help='train, val or test', default='train')  # Nathan
    parser.add_argument('--skip_imgs', action='store_true', default=False)
    parser.add_argument('--iid_pairs', action='store_true', default=False)
    parser.add_argument('--triplet', action='store_true', default=False)
    if args is None:
        args = parser.parse_args()  # uses sys.argv aka command line arguments
    else:
        args = parser.parse_args(namespace_to_arglist(args))
    return args


if __name__ == '__main__':
    npzify()