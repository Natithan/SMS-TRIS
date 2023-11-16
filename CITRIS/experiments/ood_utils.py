import random

import torch
from torch.utils import data
import wandb
from constants import CUDA_DEVICE
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

def reset_model_and_optimizer(ckpt_path, model_class, args,
                              # uf_prior_only=False,
                              unfreeze_subset=None,
                              ood_shifts=None,loaders=None, id_train_loader=None):
    model = reset_model(args, ckpt_path, model_class, ood_shifts, id_train_loader)

    optimizer, scheduler_dict = reset_optimizer(args, loaders, model, unfreeze_subset)

    return model, optimizer, scheduler_dict


def reset_model(args, ckpt_path, model_class, ood_shifts, id_train_loader):
    print("Resetting model")
    ckpt_args = {
        'checkpoint_path': ckpt_path,
        'true_ood_shifts': ood_shifts,
        'DataClass': args.data_class,
        'beta_coral': args.beta_coral,
        'strict': False
    }
    add_maybe_override_args(args, ckpt_args)
    model = model_class.load_from_checkpoint(**ckpt_args).to(CUDA_DEVICE)
    update_wandb_config(model)
    model.store_id_loss(id_train_loader, args)
    return model


def reset_optimizer(args, loaders, model, unfreeze_subset):
    print("Resetting optimizer")
    if args.coral:
        optimizer, scheduler_dict = [el[0] for el in
                                     model.configure_optimizers(max_iters=args.num_epochs * (len(loaders[('train' if 'train' in loaders else 'pretrain')])))]
        print("Unfreezing all because coral from-scratch-training")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler_dict = None

        if unfreeze_subset is not None:
            if not ("enc" in unfreeze_subset):
                print("Freezing all but prior")
                model.requires_grad_(False)
                for part in model.prior_parts:
                    part.requires_grad_(True)
    return optimizer, scheduler_dict


def add_maybe_override_args(args, ckpt_args):
    for maybe_override_arg in ['fixed_logstd', 'sample_mean']:
        if args.__dict__[maybe_override_arg] is not None:
            ckpt_args[maybe_override_arg] = args.__dict__[maybe_override_arg]
            print("Overriding", maybe_override_arg, "with", ckpt_args[maybe_override_arg])


def get_loader(args, DataClass, data_folder, split, second_data_folder=None, subset_idxs=None, mini=None, get_val_from_train=False):
    if mini is None:
        mini = args.mini
    assert (second_data_folder is None) or (subset_idxs is not None)
    shuffle = split == 'train'
    ds_kwargs = {'coarse_vars': True, 'return_latents': True}
    if all(['MixedFactorPredictor' in mc.__name__ for mc in args.model_classes]):
        ds_kwargs['require_imgs'] = False
    dl_kwargs = {'shuffle': shuffle, 'pin_memory': True, 'num_workers': args.num_workers}
    batch_size = args.batch_size if split == 'train' else args.test_batch_size
    dataset_split = split
    if dataset_split == 'val' and get_val_from_train:
        dataset_split = 'train'
    dataset = DataClass(data_folder=data_folder, split=dataset_split, **ds_kwargs, mini=mini)
    if (subset_idxs is not None) and (second_data_folder is None):
        dataset = MySubset(dataset, subset_idxs)
    if second_data_folder is None:
        dataloader = data.DataLoader(dataset, batch_size=batch_size, **dl_kwargs)
    else:
        second_dataset = DataClass(data_folder=second_data_folder, split='train', **ds_kwargs, mini=False) # Second dataset contains the few available OOD samples, which are from train even when using validation split for the first dataset
        if subset_idxs is not None:
            second_dataset = MySubset(second_dataset, subset_idxs)
        combo_dataset = FullID_FewShotOOD_Dataset(dataset, second_dataset, subset_idxs, shuffle) # __get_item__ returns a batch instead of a single sample
        RandomOrSequentialSampler = RandomSampler if shuffle else SequentialSampler
        dataloader = data.DataLoader(combo_dataset, sampler=BatchSampler(RandomOrSequentialSampler(combo_dataset),batch_size=batch_size,drop_last=False), **{
            k:dl_kwargs[k] for k in dl_kwargs if k != 'shuffle'})

    return dataloader


def update_wandb_config(model):
    from models.citris_nf import CITRISNF
    from models.citris_vae import CITRISVAE
    isCVAE = isinstance(model, CITRISVAE)
    isCNF = isinstance(model, CITRISNF)
    if isCVAE:
        if isCNF and model.skip_flow:
            ce = "SF"
        elif isCNF and model.random_flow:
            ce = "RF"
        elif model.hparams.fullview_baseline:
            if (model.hparams.no_init_masking and model.hparams.no_context_masking):
                if model.hparams.beta_classifier == 0:
                    ce = "St"
                else:
                    ce = "FV"
            else:
                ce = "Semi-FV"
        else:
            ce = "CI"
        if not isCNF:
            ce += "-VAE"
    else:
        ce = ""
    wandb.config['cfd_encouragement'] = ce


class MySubset(data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.indices = indices
        # copy the dataset's attributes
        for attr in dir(dataset):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(dataset, attr))


class FullID_FewShotOOD_Dataset(data.Dataset):
    def __init__(self, id_dataset, fs_ood_dataset, idxs, shuffle):
        self.id_dataset = id_dataset
        self.fs_ood_dataset = MySubset(fs_ood_dataset, idxs)
        self.shuffle = shuffle # at the moment, during-train_da validation still uses shuffled OOD data. Note that the this data will usually be smaller than the ID batch size anyway. It COULD be bigger, in which case this might not reach all of the OOD data. But ignoring that for now, so this shuffle flag is unused.

    def __len__(self):
        return len(self.id_dataset)

    def __getitem__(self, batch_idx):
        # batch_idx should be a list of indices
        # id_batch = [torch.stack(x) for x in zip(*[self.id_dataset[i] for i in batch_idx])]
        id_batch = {k: torch.stack([self.id_dataset[i][k] for i in batch_idx]) for k in self.id_dataset[0].keys() if k != 'isTriplet'}
        # take a random non-repeating sample of size min(batch_size, len(fs_ood_dataset)) from the ood dataset
        ood_batch_idx = random.sample(range(len(self.fs_ood_dataset)), min(len(batch_idx), len(self.fs_ood_dataset)))
        # fs_ood_batch = [torch.stack(x) for x in zip(*[self.fs_ood_dataset[i] for i in ood_batch_idx])]
        fs_ood_batch = {k: torch.stack([self.fs_ood_dataset[i][k] for i in ood_batch_idx]) for k in self.fs_ood_dataset[0].keys() if k != 'isTriplet'}
        assert len(id_batch) == len(fs_ood_batch)
        return {'id_batch': id_batch, 'ood_batch': fs_ood_batch}
