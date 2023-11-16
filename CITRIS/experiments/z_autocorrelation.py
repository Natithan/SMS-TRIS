# Nathan-added
import argparse

import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from experiments.datasets import Causal3DDataset
from utils import test_model_ckpt, get_default_parser, load_datasets
from models.citris_nf import CITRISNF
from os.path import join as jn
from constants import CITRIS_CKPT_ROOT, CITRIS_ROOT
import numpy as np
from util import plot_tensor_as_img, tn
from tqdm import tqdm

import matplotlib;

matplotlib.use("TkAgg")


def main():
    # region parse args
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # endregion
    UNCORRELATED = True
    model = CITRISNF.load_from_checkpoint(jn(CITRIS_CKPT_ROOT, "CITRISNF.ckpt"))
    model.eval()
    device = 'cuda'
    model.to(device)
    ds_folder = "/cw/liir_data/NoCsBack/TRIS/causal3d_time_dep_all7_conthue_01_coarse/"
    if UNCORRELATED:
        split='test_indep'
        also_intervened_latents = True
    else:
        split='test'
        also_intervened_latents = False
    dataset = Causal3DDataset(ds_folder, split=split, coarse_vars=True, return_latents=True,
                                  single_image=True, also_intervened_latents=also_intervened_latents)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0,
                                              pin_memory=True)  # Shuffle true or false gives little difference
    psi = model.prior_t1.get_target_assignment(hard=True)
    _, idxs = (torch.tensor([2 ** i for i in range(psi.shape[-1])][::-1])[None].to(psi.device) * psi).sum(dim=-1).sort(
        descending=True)  # Sorts rows of psi (z-dimensions) by: all z-dims belonging to first C-dim, then all z-dims belonging to second C-dim, etc.
    all_zt = []
    all_sorted_zt = []
    all_C_t = []
    for batch in tqdm(iter(test_loader)):
        with torch.no_grad():
            batch = [el.to(device) for el in batch]
            x_t, C_t = batch
            z_t = model.encode(x_t)
            sorted_zt = z_t[:, idxs]
            all_zt.append(z_t)
            all_sorted_zt.append(sorted_zt)
            all_C_t.append(C_t)
    all_zt = torch.cat(all_zt, dim=0)
    all_sorted_zt = torch.cat(all_sorted_zt, dim=0)
    all_C_t = torch.cat(all_C_t, dim=0)
    _cor_az = tn(torch.corrcoef(all_zt.t()))
    _cor_asz = tn(torch.corrcoef(all_sorted_zt.t()))
    _cor_ac = tn(torch.corrcoef(all_C_t.t()))
    if UNCORRELATED:
        non_nan_idxs = list(range(5)) + list(range(6,11))
        _cor_ac = _cor_ac[non_nan_idxs][:,non_nan_idxs] # bit hacky/nongeneral
    fig, (ax_unsorted, ax_sorted, ax_CT) = plt.subplots(ncols=3, figsize=(21, 7))
    for ax, matrix, title in zip((ax_unsorted, ax_sorted, ax_CT), (_cor_az, _cor_asz, _cor_ac),
                                 ("Unsorted", "Sorted", "C_t")):
        ax.imshow(matrix)
        ax.set_title(title)
        ax.set_aspect("equal")

        # Loop over data dimensions and create text annotations.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, matrix[i, j].round(2),
                        ha="center", va="center", color="w", size='xx-small')

    #region Annotate per-causal-factor colored boxes
    psi_s = psi[idxs]
    cmap = plt.cm.gist_rainbow
    colors = cmap(np.linspace(0, 1, psi_s.shape[-1]))
    for current_c in range(psi.shape[-1] - 1):
        a = psi_s[:, current_c].nonzero()
        start, end = a.min().item(), a.max().item()
        zgraph_end = (psi.shape[0] - 1)
        cgraph_end = (C_t.shape[0] - 1)
        z_eo = .4  # edge offset for z
        ax_sorted.add_patch(
            Rectangle((0 - z_eo, start - z_eo), zgraph_end + 2 * z_eo, end - start + 2 * z_eo, fc='none',
                      ec=colors[current_c], lw=2))
        ax_sorted.add_patch(
            Rectangle((start - z_eo, 0 - z_eo), end - start + 2 * z_eo, zgraph_end + 2 * z_eo, fc='none',
                      ec=colors[current_c], lw=2))
        if current_c == 0:
            cstart, cend = 0, 2
        elif current_c == 1:
            cstart, cend = 3, 4
        else:
            cstart = 5 + (current_c - 2)
            cend = cstart
        c_eo = .45  # edge offset for c
        ax_CT.add_patch(Rectangle(
            xy=(0 - c_eo, cstart - c_eo),
            width=cgraph_end + 2 * c_eo,
            height=cend - cstart + 2 * c_eo,
            fc='none', ec=colors[current_c], lw=2))
        ax_CT.add_patch(Rectangle(
            xy=(cstart - c_eo, 0 - c_eo),
            width=cend - cstart + 2 * c_eo,
            height=cgraph_end + 2 * c_eo,
            fc='none', ec=colors[current_c], lw=2))
    #endregion
    fig.tight_layout()
    plt.show()
    print(all_zt.shape)


if __name__ == '__main__':
    main()
