# Nathan-added
import os
import json
from subprocess import Popen
import subprocess
import numpy as np
import argparse
from constants import TC3DI_ROOT, BLENDER_PATH, IMG_GEN_SCRIPT_PATH
from data_generation.temporal_causal3dident.npzify import npzify
from experiments.datasets import Causal3DDataset
from myutil import Namespace


def get_output_folder(args
                      # ,split
                      ):
    maybe_coarse = '_coarse' if args.coarse else ''
    maybe_mini = '_mini' if args.mini else ''
    # return os.path.join(args.out_dir, f'{split}_triplets{maybe_coarse}{maybe_mini}')
    return os.path.join(args.out_dir, f'triplets{maybe_coarse}{maybe_mini}')


def create_triplet_dataset(args, split):
    """
    Generate a dataset for the triplet evaluation.
    """
    maybe_coarse = '_coarse' if args.coarse else ''
    N = args.num_samples
    output_folder = get_output_folder(args)#,split)
    start_dataset = np.load(args.start_data)
    images = start_dataset['imgs']
    nonshape_latents, shape_latents = start_dataset['raw_latents'], start_dataset['shape_latents']
    all_latents = np.concatenate([nonshape_latents, shape_latents], axis=-1)
    # targets = start_dataset['interventions']
    # has_intv = (targets.sum(axis=0) > 0).astype(np.int32)
    varies_mask = (all_latents.std(axis=0) > 0).astype(np.int32)
    varies_idxs = np.where(varies_mask)[0]
    triplet_all_latents = np.zeros((N, 3, all_latents.shape[1]), dtype=np.float32)
    prev_images = np.zeros((N, 2) + images.shape[1:], dtype=np.uint8)
    target_masks = np.zeros((N, all_latents.shape[1]), dtype=np.uint8)
    indices = np.zeros((N, 2), dtype=np.int32)
    # seed, also dependent on split
    split_to_int = {
        'train': 0,
        'val': 1,
        'test': 2,
    }
    np.random.seed(args.seed + split_to_int[split] + (10 if args.mini else 0))
    for n in range(N):
        # Pick two random images that we want to combine
        idx1 = np.random.randint(images.shape[0])
        # idx2 = np.random.randint(images.shape[0]-1)
        idx2 = np.random.randint(images.shape[0])
        # if idx2 >= idx1:
        #     idx2 += 1
        while idx1 == idx2:
            idx2 = np.random.randint(images.shape[0])
        all_latent1 = all_latents[idx1]
        all_latent2 = all_latents[idx2]
        # Pick a random combination of both images for a new, third image
        # srcs = None if has_intv.sum() > 0 else np.random.randint(2, size=(latent1.shape[0],))
        srcs = np.random.randint(2, size=(all_latent1.shape[0],)) * varies_mask
        while srcs.take(varies_idxs).astype(np.float32).std() == 0.0:
            # Prevent that we take all causal variables from one of the two images
            srcs = np.random.randint(2, size=(all_latent1.shape[0],)) * varies_mask
            # srcs = srcs * has_intv if has_intv.sum() > 0 else srcs
        all_latent3 = np.where(srcs == 0, all_latent1, all_latent2)

        triplet_all_latents[n] = np.stack([all_latent1, all_latent2, all_latent3], axis=0)
        prev_images[n,0] = images[idx1]
        prev_images[n,1] = images[idx2]
        target_masks[n] = srcs
        indices[n] = [idx1, idx2]

    # TODO save: raw_latents, shape_latents, interventions.
    # TODO in separate step, generate and save imgs
    triplet_nonshape_latents = triplet_all_latents[:,:, :nonshape_latents.shape[1]]
    triplet_shape_latents = triplet_all_latents[:,:, nonshape_latents.shape[1]:].astype(np.int32)

    os.makedirs(output_folder, exist_ok=True)

    # Save all data necessary for the triplets
    np.save(os.path.join(output_folder, f'full_latents_{split}.npy'), triplet_nonshape_latents)
    np.save(os.path.join(output_folder, f'full_shape_latents_{split}.npy'), triplet_shape_latents)
    np.save(os.path.join(output_folder, f'interventions_{split}.npy'), target_masks) # They're not actually interventions, but we use this name for compatiblity with npzipify and later dataset loading
    np.save(os.path.join(output_folder, f'sources_indices_{split}.npy'), indices)
    np.save(os.path.join(output_folder, f'prev_images_{split}.npy'), prev_images)
    # Save the latents for blender to generate the new images
    np.save(os.path.join(output_folder, f'latents_{split}.npy'), triplet_nonshape_latents[:,-1])
    np.save(os.path.join(output_folder, f'shape_latents_{split}.npy'), triplet_shape_latents[:,-1])

    hparams = {}
    hparams['orig_dataset'] = args.start_data
    hparams['triplets'] = {
        'coarse_vars': args.coarse,
        'num_triplet_points': N,
        'orig_dataset': args.start_data
    }
    hparams['seed'] = args.seed
    with open(os.path.join(output_folder, f'hparams_{split}.json'), 'w') as f:
        json.dump(hparams, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--out_dir', type=str, default=TC3DI_ROOT, help='Output directory')
    parser.add_argument('--start_data', type=str, default=os.path.join(TC3DI_ROOT, 'train.npz'), help='Start dataset')
    parser.add_argument('--mini', action='store_true', help='Use and create mini dataset')
    parser.add_argument('--coarse', type=bool, default=False, help='Generate coarse dataset')
    parser.add_argument('--overwrite_imgs', action='store_true', help='Override existing images')
    parser.add_argument('--n_parallel', type=int, default=20, help='Number of parallel processes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    maybe_mini = '_mini' if args.mini else ''
    if args.mini:
        args.num_samples = min(args.num_samples, 1000)
    args.start_data = args.start_data.replace('.npz', f'{maybe_mini}.npz')
    maybe_coarse = '_coarse' if args.coarse else ''
    splits = [
        'train',
        'val',
        'test'
    ]
    for split in splits:
        if split in ['val', 'test']:
            args.num_samples = min(args.num_samples, 10000)
        create_triplet_dataset(args, split)

    for split in splits:
        output_folder = get_output_folder(args)#, split)
        print(f"n_parallel: {args.n_parallel}")

        processes = [Popen(
            f"{BLENDER_PATH} --background --python {IMG_GEN_SCRIPT_PATH} -- --output-folder {output_folder} --n-batches {args.n_parallel} --batch-index {batch_index} --split {split}" + (" --overwrite" if args.overwrite_imgs else ""),
            shell=True, stderr=subprocess.PIPE) for batch_index in range(args.n_parallel)]
        for p in processes:
            p.wait()


        npzify_args = Namespace(dir=output_folder, split=split, triplet=True)
        npzify(npzify_args)