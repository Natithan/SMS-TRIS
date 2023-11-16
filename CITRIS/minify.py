import numpy as np
from constants import TC3DI_ROOT

files = ["train", "test", "test_indep", "test_triplets_coarse"]
paths = [f"{TC3DI_ROOT}/{t}.npz" for t in files]
mini_paths = [f"{TC3DI_ROOT}/{t}_mini.npz" for t in files]

mini_size = 1000
for p, mp in zip(paths, mini_paths):
    arr = np.load(p)
    mini_arr = [arr[f][:mini_size] for f in arr.files]
    np.savez(mp, **{f: mini_arr[i] for i, f in enumerate(arr.files)})