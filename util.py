from os.path import join as jn
from glob import glob
from constants import CITRIS_CKPT_ROOT
def plot_tensor_as_img(tensor,savepath=None, title=None):
    if savepath is None:
        import matplotlib;
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    t = fix_tensor(tensor)
    im = plt.imshow(t)
    if title is not None:
        plt.title(title)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)


def fix_tensor(tensor,vid=False):
    import numpy as np
    # Check if type is numpy
    if type(tensor) != np.ndarray:
        t = tensor.detach().cpu().numpy()
    else:
        t = tensor
    if not vid:
        if t.shape[0] == 3:
            t = t.transpose(1, 2, 0)
    else:
        if t.shape[1] == 3:
            t = t.transpose(0, 2, 3, 1)
    if -1.1 < t.min() < -.9 and .6 < t.max() < 1.1:
        print("Scaling from [-1,1] to [0,1]")
        t = (t + 1) / 2
    if vid:
        t = (t * 255).astype(np.uint8)
    return t


def plot_tensors_as_stacked_imgs(*tensors, horizontal=True, savepath=None, title=None):
    '''
    Plots list of batch tensors as 2D image grid. If horizontal is True, the images are stacked horizontally along the list, and vertically along the batch dimension.
    If vertical is True, vice versa.
    Also works for non-batch tensors.
    '''
    import torch
    if len(tensors[0].shape) == 3:
        tensors = [t[None] for t in tensors]
    stacked_tensor = torch.concat([t.permute(1, 2, 0, 3).flatten(-2, -1) for t in tensors], dim=1 if horizontal else 2) # Didn't test vertical :P
    plot_tensor_as_img(stacked_tensor, savepath=savepath, title=title)


def plot_tensor_as_vid(tensor,pause_time=.2, savepath=None):
    if savepath is None:
        import matplotlib;
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    tensor = fix_tensor(tensor,vid=True)
    if savepath is None:
        first_frame = tensor[0]
        im = plt.imshow(first_frame)
        for frame in tensor:
            im.set_data(frame)
            plt.pause(pause_time)
            plt.show()
    else:
        import imageio
        imageio.mimsave(savepath, tensor)


def tn(t):
    return t.detach().cpu().numpy()


def count_params(model,trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def print_param_count(pl_module, prefix = ""):
    print(prefix,f'All: {count_params(pl_module):,}',f'Trainable: {count_params(pl_module,trainable_only=True):,}')

def summ(tensor):
    import numpy as np
    np_tensor = tn(tensor) if type(tensor) != np.ndarray else tensor
    print("shape: ", np_tensor.shape)
    print("min/max: ", np_tensor.min(), np_tensor.max())
    print("mean: ", np_tensor.mean())
    print("std: ", np_tensor.std())


def print_minmax_multiindex(tensor, n=1):
    import numpy as np
    np_tensor = tn(tensor) if type(tensor) != np.ndarray else tensor
    # print("min/max: ", np_tensor.min(), np_tensor.max())
    # print("argmin/argmax: ", np.unravel_index(np_tensor.argmin(), np_tensor.shape), np.unravel_index(np_tensor.argmax(), np_tensor.shape))
    print("min")
    # smallest n values
    for i in np.argpartition(np_tensor.flatten(), n)[:n]:
        print(np.unravel_index(i, np_tensor.shape), np_tensor.flatten()[i])
    print("max")
    # largest n values
    for i in np.argpartition(np_tensor.flatten(), -n)[-n:]:
        print(np.unravel_index(i, np_tensor.shape), np_tensor.flatten()[i])


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def CUDAfy(batch):
    CUDA_DEVICE = 'cuda'
    # if type(batch) == dict:
    #     # dict of lists of tensors to cuda
    #     batch = {k: [el.to(CUDA_DEVICE) for el in v] for k, v in batch.items()}
    # else:
    #     batch = [el.to(CUDA_DEVICE) if el != [] else [] for el in batch]  # can be empty list if require_imgs is False
    if type(list(batch.values())[0]) == dict:
        # dict of dicts of tensors to cuda
        batch = {k: {k2: el2.to(CUDA_DEVICE) for k2, el2 in v.items()} for k, v in batch.items()}
    else:
        batch = {k: el.to(CUDA_DEVICE) if el != [] else [] for k, el in batch.items()}
    return batch

def get_ckptpath_for_wid(wid):
    path = glob(jn(CITRIS_CKPT_ROOT, f"{wid}_*/epoch=*-step=*.ckpt"))[0]
    assert path is not None, f"Couldn't find path for wid {wid}"
    return path

def mylog(self, name, value):
    '''
    Hack to log to multiple dataloaders # TODO I think simpler solution is to add argument add_dataloader_idx=False to log
    '''
    dataloader_idx = self.trainer._results.dataloader_idx
    self.trainer._results.dataloader_idx = None
    self.log(name, value)
    self.trainer._results.dataloader_idx = dataloader_idx