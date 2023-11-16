import glob
from os.path import join as j
import os
from constants import TC3DI_OOD_ROOT, PONG_OOD_ROOT
import numpy as np
import imageio
from tqdm import tqdm
import multiprocessing as mp

for subfolder in next(os.walk(TC3DI_OOD_ROOT))[1]:
    for split in ['train', 'val', 'test']:
        maybe_unordered_glob = glob.glob(j(TC3DI_OOD_ROOT, subfolder, f'images_{split}', '*.png'))
        ordered_glob = sorted(glob.glob(j(TC3DI_OOD_ROOT, subfolder, f'images_{split}', '*.png')))
        # if maybe_unordered_glob != ordered_glob:
        #     print(f'Unordered glob for {subfolder} {split}!')
        #     print(f"Number of items out of place: {sum([1 for i in range(len(maybe_unordered_glob)) if maybe_unordered_glob[i] != ordered_glob[i]])} out of {len(maybe_unordered_glob)} (fraction: {sum([1 for i in range(len(maybe_unordered_glob)) if maybe_unordered_glob[i] != ordered_glob[i]]) / len(maybe_unordered_glob)})")
        #     print(f"First item not matching ordered glob: {[i for i in range(len(maybe_unordered_glob)) if maybe_unordered_glob[i] != ordered_glob[i]][0]}")
        #     unordered_digits = [int(os.path.basename(f).split('.')[0]) for f in maybe_unordered_glob]
        #     print(f"Number of items followed by an incorrect item: {sum([1 for i in range(len(unordered_digits)-1) if unordered_digits[i] != unordered_digits[i+1]-1])}")
        # else:
        #     print(f'Ordered glob for {subfolder} {split}!')

        maybe_unordered_npz = np.load(j(TC3DI_OOD_ROOT, subfolder, f'{split}.npz'))
        maybe_unordered_npz_imgs = maybe_unordered_npz['imgs']
        # ordered_npz_imgs = np.stack([imageio.v2.imread(impath) for impath in ordered_glob])
        # to_stack = []
        # for impath in tqdm(ordered_glob):
        #     to_stack.append(imageio.v2.imread(impath))
        # ordered_npz_imgs = np.stack(to_stack) # Make this more efficient with parallelism
        max_workers = mp.cpu_count()
        with mp.Pool(max_workers) as p:
            ordered_npz_imgs = np.stack(p.map(imageio.v2.imread, ordered_glob))

        matching_elements = (maybe_unordered_npz_imgs == ordered_npz_imgs)
        matching_images = np.all(matching_elements, axis=(1, 2, 3))
        if not np.all(matching_images):
            print(f'Unordered npz for {subfolder} {split}!')
            print(f"Number of images out of place: {sum([1 for el in matching_images if not el])} out of {len(matching_images)} (fraction: {sum([1 for el in matching_images if not el]) / len(matching_images)})")
            print(f"First image not matching ordered npz: {[i for i in range(len(matching_images)) if not matching_images[i]][0]}")
        else:
            print(f'Ordered npz for {subfolder} {split}!')