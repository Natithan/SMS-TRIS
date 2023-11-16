# Nathan-added
import argparse

import torch
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from experiments.datasets import Causal3DDataset
from utils import test_model_ckpt, get_default_parser, load_datasets
from models.citris_nf import CITRISNF
from os.path import join as jn
from constants import CITRIS_CKPT_ROOT, CITRIS_ROOT, DATASET_ROOT_FOLDER, TC3DI_ROOT
import numpy as np
from util import plot_tensor_as_img
from tqdm import tqdm

OOD_PATH = jn(CITRIS_ROOT, "data_generation/temporal_causal3dident/1000_points_x_pos/")
ID_PATH = TC3DI_ROOT

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    og_model = CITRISNF.load_from_checkpoint(jn(CITRIS_CKPT_ROOT, "CITRISNF.ckpt"))
    fullview_model = CITRISNF.load_from_checkpoint(jn(CITRIS_CKPT_ROOT, "CITRISNF_fullview.ckpt"))
    mselosses = {}
    for model,name in zip((og_model, fullview_model), ("CITRISNF", "FVTRISNF")):
        model.eval()
        device = 'cuda'
        model.to(device)
        mselosses[name] = {}
        for ds_folder,ds_name in (
                (ID_PATH, 'ID'),
                (OOD_PATH,'OOD')
        ):
            dataset = Causal3DDataset(ds_folder, split='test', coarse_vars=True, return_latents=True)
            ds_name += f" ({len(dataset)} points)"
            mselosses[name][ds_name] = []
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=0,pin_memory=True)

            for batch in tqdm(iter(test_loader)):
                with torch.no_grad():
                    batch = [el.to(device) for el in batch]
                    x_t, x_t1, I_t1, C_t, C_t1 = batch[0][:, 0], batch[0][:, 1], batch[1], batch[2][:, 0], batch[2][:, 1]
                    z_t = model.encode(x_t)
                    z_t1_pred = model.sample_zt1(z_t, I_t1)
                    x_t1_pred = model.autoencoder.decoder(model.flow.reverse(z_t1_pred))
                    mselosses[name][ds_name].append(torch.mean((x_t1 - x_t1_pred) ** 2))
                    # plot_tensor_as_img(
                    #     torch.concat([x.permute(1, 2, 0, 3).flatten(-2, -1) for x in (x_t1, x_t1_pred, x_t1_reconstr)], dim=1))
                    # break
            mselosses[name][ds_name] = torch.stack(mselosses[name][ds_name]).mean()
    matrix = np.zeros((2, 2))
    for i, (k, v) in enumerate(mselosses.items()):
        for j, (k2, v2) in enumerate(v.items()):
            matrix[i, j] = v2.item()
    matplotlib.use('TkAgg')
    plt.imshow(matrix, cmap='Greys')
    plt.yticks(range(2), list(mselosses.keys()),fontsize=15)
    plt.xticks(range(2), mselosses[list(mselosses.keys())[0]].keys(),fontsize=15)
    # Loop over data dimensions and create text annotations.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, matrix[i, j].round(4),
                    ha="center", va="center", color="green", size='xx-large')

    plt.show()
    print(mselosses)


if __name__ == '__main__':
    main()