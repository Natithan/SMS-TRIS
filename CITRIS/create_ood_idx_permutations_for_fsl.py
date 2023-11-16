import argparse
import os
import numpy as np
import json
from constants import PONG_OOD_ROOT, TC3DI_OOD_ROOT
import re

from myutil import namespace_to_arglist


def create_idx_permutations(args = None):
    # Take args: path to data superfolder, list of numbers of shots
    parser = argparse.ArgumentParser()
    parser.add_argument('--ood_data_dirs', type=str, nargs='+', default=[PONG_OOD_ROOT, TC3DI_OOD_ROOT])
    parser.add_argument('--n_permutations', type=int, default=100)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--n_points', type=int, default=-1)
    if args is None:
        args = parser.parse_args()  # uses sys.argv aka command line arguments
    else:
        args = parser.parse_args(namespace_to_arglist(args))
    for ood_data_dir in args.ood_data_dirs:
        if args.n_points != -1:
            n_ood_points = args.n_points
        else:
            n_ood_points = int(re.search(r'/(\d+)_', ood_data_dir).group(1)) # the 1 indicates the first group in the regex (otherwise would return the whole match)
        n_ood_pairs = n_ood_points - 1 # Because targets are shifted by 1
        seed2idx = {}
        for seed in range(args.n_permutations):
            split_seed = seed + args.n_permutations if args.split == 'val' else seed
            np.random.seed(split_seed)
            seed2idx[seed] = np.random.choice(n_ood_pairs, n_ood_pairs, replace=False).tolist()
        # # Save dict to json
        # with open(os.path.join(ood_data_dir, 'idx_permutations.json'), 'w') as f:
        #     json.dump(seed2idx, f)
        # Save dict to json if not exists already, else print warning
        if not os.path.exists(os.path.join(ood_data_dir, f'idx_permutations_{args.split}.json')):
            with open(os.path.join(ood_data_dir, f'idx_permutations_{args.split}.json'), 'w') as f:
                json.dump(seed2idx, f)
        else:
            print(f"WARNING: idx_permutations_{args.split}.json already exists for {ood_data_dir}, not overwriting.")



if __name__ == '__main__':
    create_idx_permutations()