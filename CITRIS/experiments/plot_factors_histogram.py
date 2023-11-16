# region imports
import math
import warnings
warnings.filterwarnings("ignore")
from constants import CITRIS_ROOT, DATASET_ROOT_FOLDER, TC3DI_ROOT, TC3DI_OOD_ROOT
from experiments.datasets import Causal3DDataset
SHAPES_FACTORS = Causal3DDataset.FACTORS
SHAPES_F2SHORT = Causal3DDataset.F2SHORT
from os.path import join as jn
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from util import tn
import numpy as np
import os
from mpl_toolkits.axes_grid1 import axes_grid
from data_generation.temporal_causal3dident.data_generation_causal3dident import ID_F2P, OOD_F2P
from tqdm import tqdm
import argparse
# endregion


def main():
    # region parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--ood_kind', type=str, default='mech_shift', choices=['mech_shift', 'decorr_source_frames'])
    args = parser.parse_args()
    # endregion
    # all_t_for_t1 = {k:SHAPES_FACTORS for k in shifted_factors}
    # t_for_t1 = all_t_for_t1

    iid_data_dir = TC3DI_ROOT
    iid_train_dataset = get_ds(iid_data_dir, "train")
    if args.ood_kind == 'mech_shift':
        iid_test_dataset = get_ds(iid_data_dir, 'test')
        plt_mech_shifts_histograms(iid_test_dataset, iid_train_dataset)
    elif args.ood_kind == 'decorr_source_frames':
        plt_decorr_source_frames_histograms(iid_train_dataset)
    plt.savefig(f'factors_histogram_{args.ood_kind}.png')
    print(5)


def plt_mech_shifts_histograms(iid_test_dataset, iid_train_dataset):
    n_rows = len(SHAPES_FACTORS)
    n_cols = len(ID_F2P)

    fig = plt.figure(figsize=(20*n_rows,5*n_cols))
    outer_grid = fig.add_gridspec(n_rows, n_cols, wspace=0.1, hspace=0.1)
    shifted_factors = ID_F2P.keys()
    for col_idx, shifted_factor in tqdm(zip(range(n_cols), shifted_factors)):

        # ood_data_dir = jn(CITRIS_ROOT, f"data_generation/temporal_causal3dident/1000_points_{shifted_factor}")
        ood_data_dir = jn(TC3DI_OOD_ROOT, f"{shifted_factor}")

        all_ood_datasets = [get_ds(ood_data_dir, 'train'), get_ds(ood_data_dir, 'val'), get_ds(ood_data_dir, 'test')]
        for row_idx, t1_factor in zip(range(n_cols), SHAPES_FACTORS):
            t1_id = SHAPES_FACTORS.index(t1_factor)
            t_factors = ID_F2P[t1_factor]
            t_ids = [SHAPES_FACTORS.index(f) for f in t_factors]

            iid_train_t1_latents, iid_train_t_latents_list = get_t_t1_latents(iid_train_dataset, t1_id, t_ids)
            # Limit to 10000 samples
            iid_train_t1_latents = iid_train_t1_latents[:10000]
            iid_train_t_latents_list = [t_latents[:10000] for t_latents in iid_train_t_latents_list]
            iid_test_t1_latents, iid_test_t_latents_list = get_t_t1_latents(iid_test_dataset, t1_id, t_ids)

            ood_t1_latents_combined = np.concatenate([get_t1_latents(ds, t1_id) for ds in all_ood_datasets])
            ood_t_latents_combined = [np.concatenate([get_t_latents(ds, t_id) for ds in all_ood_datasets]) for t_id in
                                      t_ids]
            inner_grid = outer_grid[row_idx, col_idx].subgridspec(3, 1, wspace=0, hspace=0)
            axs = inner_grid.subplots()
            for j, (t1_latents, t_latents_list) in enumerate(
                    zip([iid_train_t1_latents, iid_test_t1_latents, ood_t1_latents_combined],
                        [iid_train_t_latents_list, iid_test_t_latents_list, ood_t_latents_combined])):
                ax = axs[j]
                cond_hist_array = get_cond_hist_array(t1_latents, t_latents_list)
                ax.imshow(cond_hist_array)
                # hide ticks
                ax.set_xticks([])
                ax.set_yticks([])
            if shifted_factor == t1_factor:
                red_outline_multiaxs(axs)
            # set labels
            # if row_idx == 0:
            # Set title above ax
            axs[0].set_title('OOD_' + SHAPES_F2SHORT[shifted_factor], loc='left')
            # if col_idx == 0:
            axs[1].set_ylabel('t1: ' + SHAPES_F2SHORT[t1_factor])
            axs[2].set_xlabel('t: ' + ', '.join([SHAPES_F2SHORT[f] for f in t_factors]))

def plt_decorr_source_frames_histograms(iid_train_dataset):
    n_rows = n_cols = len(SHAPES_FACTORS)
    fig = plt.figure(figsize=(100,80))
    outer_grid = fig.add_gridspec(n_rows, n_cols, wspace=0.1, hspace=0.1)
    decorr_train_dataset = get_ds(TC3DI_ROOT, 'val',decorr_source_frames=True)
    for row_idx, t1_factor in tqdm(zip(range(n_rows), SHAPES_FACTORS)):
        for col_idx, t_factor in zip(range(n_cols), SHAPES_FACTORS):
            inner_grid = outer_grid[row_idx, col_idx].subgridspec(2, 1, wspace=0, hspace=0)
            axs = inner_grid.subplots()
            for ds, ax in zip([iid_train_dataset, decorr_train_dataset], axs):
                t1_id = SHAPES_FACTORS.index(t1_factor)
                t_id = SHAPES_FACTORS.index(t_factor)
                t1_latents = get_t1_latents(ds, t1_id)[:10000]
                t_latents = get_t_latents(ds, t_id)[:10000]
                cond_hist_array = get_cond_hist_array(t1_latents, [t_latents])
                ax.imshow(cond_hist_array)
                # hide ticks
                ax.set_xticks([])
                ax.set_yticks([])
            if t_factor in ID_F2P[t1_factor]:
                red_outline_multiaxs(axs)
            # set labels
            axs[0].set_ylabel('t1: ' + SHAPES_F2SHORT[t1_factor])
            axs[1].set_xlabel('t: ' + SHAPES_F2SHORT[t_factor])
    plt.show()


def get_cond_hist_array(t1_latents, t_latents_list, n_bins=10):
    joint_data = (np.stack([t1_latents] + t_latents_list)).transpose()
    hist_array, _ = np.histogramdd(joint_data, bins=n_bins)
    marginal_hist_array = hist_array.sum(axis=0)[None]
    cond_hist_array = hist_array / marginal_hist_array
    cond_hist_array = cond_hist_array.reshape(n_bins, math.prod([n_bins for _ in t_latents_list]))
    return cond_hist_array


def red_outline_multiaxs(axs):
    # Draw red outline
    for i, ax in enumerate(axs):
        for loc in ['top', 'bottom', 'left', 'right']:
            if i == 0:
                if loc == 'bottom':
                    continue
            elif i == 1:
                if loc in ['top', 'bottom']:
                    continue
            elif i == 2:
                if loc == 'top':
                    continue
            ax.spines[loc].set_color('red')
            ax.spines[loc].set_linewidth(5)


def get_t_t1_latents(dataset, t1_id, t_ids):
    t_latents_list = [get_t_latents(dataset, t_id) for t_id in t_ids]
    t1_latents = get_t1_latents(dataset, t1_id)
    return t1_latents, t_latents_list

def get_t_latents(dataset, t_id):
    return tn(dataset.true_latents[:-1, t_id]) if len(dataset.true_latents.shape) == 2 else tn(dataset.true_latents[:, 0, t_id])

def get_t1_latents(dataset, t1_id):
    return tn(dataset.true_latents[1:, t1_id]) if len(dataset.true_latents.shape) == 2 else tn(dataset.true_latents[:, 1, t1_id])

def get_ds(data_folder, split, decorr_source_frames=False):
    return Causal3DDataset(data_folder=data_folder, split=split, coarse_vars=True, return_latents=True, cma=decorr_source_frames,
                           causal_vars=['pos-x', 'pos-y', 'pos-z', 'rot-alpha', 'rot-beta', 'rot-spot', 'hue-object', 'hue-spot', 'hue-back', 'obj-shape']) # For the normal continuous dataset, causal_vars is extracted from the interventions. In the train-dataloading, the causal vars for non-continuous datasets are then set to the causal vars of the continuous dataset. Here however, we specify the causal vars manually.

if __name__ == '__main__':
    main()
