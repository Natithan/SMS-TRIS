import torch

from experiments.datasets import Causal3DDataset

from constants import CITRIS_ROOT, DATASET_ROOT_FOLDER, TRIS_ROOT

from models.shared.transition_prior import TrueLatentPrior
from os.path import join as jn
import numpy as np
import matplotlib.pyplot as plt
# Test if true latent prior with setting old gives same output as without

def main():
    data_dir = jn(TRIS_ROOT,"causal3d_time_dep_all7_conthue_01_coarse")
    dataset = Causal3DDataset(data_folder=data_dir, split='test', coarse_vars=True, return_latents=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=5,
                                   shuffle=True, pin_memory=True, num_workers=0)

    old_model = TrueLatentPrior(num_causal_vars=7, c_hid=64, c_out=2, true_ood_shifts=None, only_parents=True,
                        nonfv_target_layer=False, old=True, nokaiming=False, data_folder="causal3d_time_dep_all7_conthue_01_coarse")
    new_model = TrueLatentPrior(num_causal_vars=7, c_hid=64, c_out=2, true_ood_shifts=None, only_parents=True,
                        nonfv_target_layer=False, old=False, nokaiming=False, data_folder="causal3d_time_dep_all7_conthue_01_coarse")

    match_MV_to_nonMV(new_model, old_model)

    batch = next(iter(loader))
    old_model.eval()
    new_model.eval()
    old_loss = old_model.get_per_c_loss(batch)
    new_loss = new_model.get_per_c_loss(batch)
    print(old_loss)
    print(new_loss)
    print(5)


def match_MV_to_nonMV(new_model, old_model):
    # Make sure weights are the same using named parameters
    for (old_name, old_param), (new_name, new_param) in zip(old_model.named_parameters(), new_model.named_parameters()):
        # print(old_name, new_name)
        assert old_name == new_name

        if old_name.startswith('context_layer') or old_name.startswith('target_layer'):
            new_param.data = old_param.data.unflatten(0, (10, -1))
        else:
            new_param.data = old_param.data


if __name__ == '__main__':
    main()