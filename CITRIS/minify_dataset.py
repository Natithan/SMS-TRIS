import argparse
import numpy as np
import os

MINI_LENS = {
    'train': 1024,
    'val': 1024,
    'test': 1024,
    'val_indep': 1024,
    'test_indep': 1024,
    'val_triplets': 1024,
    'test_triplets': 1024,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()

    npz_objs = {f.split('.npz')[0]: np.load(f'{args.data_dir}/{f}') for f in os.listdir(args.data_dir) if
                f.endswith('.npz') and ('mini' not in f)}
    for f in npz_objs:
        new_obj = {}
        for key in npz_objs[f].files:
            # print(f,key,len(npz_objs[f][key]))
            max_idx = MINI_LENS[f]
            if key == 'targets' and not any([s in f for s in ['triplets', 'indep']]):
                max_idx -= 1
            new_obj[key] = npz_objs[f][key][:max_idx]
            print(f,key,len(npz_objs[f][key]), '->', len(new_obj[key]))
        np.savez_compressed(os.path.join(args.data_dir,f'{f}_mini.npz'), **new_obj)


if __name__ == '__main__':
    main()
