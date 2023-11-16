import jsonargparse

from experiments.train_nf import encode_datasets
from pretorch_util import assign_visible_gpus
assign_visible_gpus(free_mem_based=True)
from myutil import add_id_ood_common_args, add_id_common_args, add_flow_args
from experiments.datasets import get_DataClass_for_datadir
from experiments.utils import train_model, maybe_minify_args, update_aeckpt_arg
from models.shared import AutoregNormalizingFlow
from torch.utils.data import DataLoader


def main():

    # region parse args
    parser = jsonargparse.ArgumentParser(default_config_files=['cfgs/Flow_ppt_shapes.yml']) #default_config_files=['cfgs/cheat_cfd.yml']) is empty for now
    add_id_ood_common_args(parser)
    add_id_common_args(parser)
    add_flow_args(parser)
    add_custom_args(parser)

    args = parser.parse_args()
    maybe_minify_args(args)
    update_aeckpt_arg(args)
    # endregion

    DataClass = get_DataClass_for_datadir(args.data_dir)
    # region datasets and loaders
    common_ds_kwargs = dict(
        single_image=True,
        # seq_len=1, is set to 1 anyway if single_image=True
        mini=args.mini,
        data_folder=args.data_dir,
        return_latents=True,
    )
    train_single_dataset = DataClass(split='train', **common_ds_kwargs)
    val_single_dataset = DataClass(split='val', **common_ds_kwargs)
    test_single_dataset = DataClass(split='test', **common_ds_kwargs)
    triplet_ds_kwargs = common_ds_kwargs | {'single_image':False, 'triplet':True, 'causal_vars':train_single_dataset.full_target_names, "coarse_vars": False}
    if args.subtest:
        triplet_ds_kwargs['subtest'] = True
    test_triplet_dataset = DataClass(**triplet_ds_kwargs | {'split':'test'})
    val_triplet_dataset = DataClass(**triplet_ds_kwargs | {'split':'val'})
    if args.triplet_train:
        # if not args.subtest:
        #     raise NotImplementedError("triplet_train only implemented for subtest")
        train_triplet_dataset = DataClass(**triplet_ds_kwargs | {'split':'train'})
    common_corr_ds_kwargs = common_ds_kwargs | {
        'causal_vars': train_single_dataset.full_target_names
    }
    val_corr_dataset = DataClass(**(common_corr_ds_kwargs | {'split': 'val_indep'}))
    test_corr_dataset = DataClass(**(common_corr_ds_kwargs | {'split': 'test_indep'}))
    datasets_to_encode = [train_single_dataset, val_single_dataset, test_single_dataset, val_corr_dataset, test_corr_dataset,
     val_triplet_dataset, test_triplet_dataset] + ([train_triplet_dataset] if args.triplet_train else [])
    common_dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_single_loader = DataLoader(train_single_dataset, shuffle=True, drop_last=True, **common_dl_kwargs)
    val_single_loader = DataLoader(val_single_dataset, shuffle=False, drop_last=False, **common_dl_kwargs)
    test_single_loader = DataLoader(test_single_dataset, shuffle=False, drop_last=False, **common_dl_kwargs)

    test_triplet_loader = DataLoader(test_triplet_dataset, shuffle=False,
                                          drop_last=False, **common_dl_kwargs)
    val_triplet_loader = DataLoader(val_triplet_dataset, shuffle=False,
                                            drop_last=False, **common_dl_kwargs)
    if args.triplet_train:
        train_triplet_loader = DataLoader(train_triplet_dataset, shuffle=True,
                                            drop_last=True, **common_dl_kwargs)
    train_loader = train_triplet_loader if args.triplet_train else train_single_loader
    # endregion
    logger_name = f'{"mini_" if args.mini else ""}Flow_cheat_cfd'
    train_model(model_class=AutoregNormalizingFlow,
                train_loader=train_loader,
                val_loaders_dict={'z2c':val_single_loader, 'triplet': val_triplet_loader},
                test_loaders_dict={'z2c': test_single_loader, 'triplet': test_triplet_loader},
                logger_name=logger_name,
                DataClass=DataClass,
                op_before_running=lambda model: encode_datasets(model, datasets_to_encode, data_folder=[s for s in args.data_dir.split('/') if len(s) > 0][-1]),
                callback_kwargs={'correlation_dataset': val_corr_dataset,
                                 'correlation_test_dataset': test_corr_dataset,
                                 'mini': args.mini,
                                 'skip_correnc_train': args.skip_correnc_train,
                                 },
                val_track_metric='val_comb_loss',
                **vars(args))


def add_custom_args(parser):
    parser.add_argument('-c', '--config', help="configuration file *.yml. Can be overriden with direct args",
                        action=jsonargparse.ActionConfigFile)
    parser.add_argument('--angle_reg_weight', type=float, default=0.1)
    # parser.add_argument('--grad_reverse', action='store_true')
    parser.add_argument('--lambda_grad_reverse', type=float, default=0.0)
    # parser.add_argument('--zero_nonmatch', action='store_true')
    # parser.add_argument('--straightthru_nonmatch', action='store_true')
    parser.add_argument('--nonmatch_strategy', type=str, default='gradreverse')
    parser.add_argument('--triplet_train', action='store_true')
    parser.add_argument('--subtest', action='store_true')
    parser.add_argument('--z2c_triplet_ratio', type=float, default=0.5)
    parser.add_argument('--skip_correnc_train', action='store_true')
    parser.add_argument('--nd_as_loss', action='store_true')
    parser.add_argument('--lambda_reg_z', type=float, default=0.0)

if __name__ == '__main__':
    main()
