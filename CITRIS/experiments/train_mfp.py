"""
Run file for training TrueLatentPrior: a model that needs to predict next-timestep-Cs based on previous-timestep-Cs and I.
"""
import jsonargparse

from models.mixed_factor_predictor import MixedFactorPredictor
from pretorch_util import assign_visible_gpus
assign_visible_gpus(free_mem_based=True)
import sys
from myutil import add_id_ood_common_args, add_id_common_args

sys.path.append('../')
from experiments.utils import train_model, load_datasets, get_default_parser, print_params, maybe_minify_args


def add_specific_args(parser):
    parser.add_argument('-c', '--config', help="configuration file *.yml. Can be overriden with direct args",
                        action=jsonargparse.ActionConfigFile)
    parser.add_argument('--exclude_vars', type=str, nargs='+', default=None)
    parser.add_argument('--exclude_objects', type=str, nargs='+', default=None)
    parser.add_argument('--coarse_vars', action='store_true')
    parser.add_argument('--data_img_width', type=int, default=-1)
    parser.add_argument('--seq_len', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--logger_name', type=str, default='')
    parser.add_argument('--files_to_save', type=str, nargs='+', default=None)
    parser.add_argument('--c_hid', type=int, default=64)
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--hidden_per_var', type=int, default=16)
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--autoregressive_prior', action='store_true')
    parser.add_argument('--include_og_imgs', type=bool, default=False)
    parser.add_argument('--only_parents', type=bool, default=False)
    parser.add_argument('--nonfv_target_layer', action="store_true")
    parser.add_argument('--old', action="store_true")
    parser.add_argument('--nokaiming', action="store_true")
    parser.add_argument('--fixed_encoder', type=str, default=None, help="Either 'perfect_ux' or 'identity")
    parser.add_argument('--cma_true_parentage', action="store_true", help="Only if --fixed_encoder=perfect_ux: mask inputs to the FLP that are not ground truth parents ('cheat' to know ground truth parents)")
    parser.add_argument('--encourage_cfd', action="store_true")


if __name__ == '__main__':
    # region Parse args
    parser = jsonargparse.ArgumentParser(default_config_files=['cfgs/MFP.yml'])
    add_specific_args(parser)
    add_id_common_args(parser)
    add_id_ood_common_args(parser)
    args = parser.parse_args()
    # endregion
    maybe_minify_args(args)
    args.return_latents_train = True
    args.require_imgs = False # Not needed for MFP setting
    model_args = vars(args)
    if args.no_intvs:
        args.data_dir = args.data_dir[:-1] + '_no_intvs/'
    datasets, data_loaders, data_name = load_datasets(args)
    DataClass = datasets['train'].__class__
    model_args['DataClass'] = DataClass
    model_args['max_iters'] = args.max_epochs * len(data_loaders['train'])
    model_args['num_causal_vars'] = datasets['train'].num_vars()
    model_class = MixedFactorPredictor

    # logger_name = args.prename + ('mini_' if args.mini else '') + (f'{"P" if args.fixed_encoder == "perfect_ux" else "I" if args.fixed_encoder == "identity" else "" }MFP' + args.postname)
    logger_name = ('mini_' if args.mini else '') + f'{"P" if args.fixed_encoder == "perfect_ux" else "I" if args.fixed_encoder == "identity" else ""}MFP'
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name

    print_params(logger_name, model_args)
    
    check_val_every_n_epoch = model_args.pop('check_val_every_n_epoch')
    if check_val_every_n_epoch <= 0:
        check_val_every_n_epoch = 5
    train_model(model_class=model_class,
                train_loader=data_loaders['train' if not args.train_on_decorr else 'train_cma_tlp'],
                val_loaders_dict={'tlp': data_loaders['val_tlp'], 'cma_tlp': data_loaders['val_cma_tlp']},
                test_loaders_dict={'tlp': data_loaders['test_tlp'], 'cma_tlp': data_loaders['test_cma_tlp']},
                logger_name=logger_name,
                check_val_every_n_epoch=check_val_every_n_epoch,
                progress_bar_refresh_rate=1,
                # callback_kwargs={'dataset': datasets['train'],
                #                  'correlation_dataset': datasets['val'],
                #                  'correlation_test_dataset': datasets['test'],
                #                  'add_enco_sparsification': args.enco_postprocessing,
                #                  'mini': args.mini},
                callback_kwargs={'dataset': datasets['train'],
                                 'add_enco_sparsification': args.enco_postprocessing,
                                 'mini': args.mini} |
                                 ({'correlation_dataset': datasets['val_corr'], 'correlation_test_dataset': datasets['test_corr']} if not args.no_alignment_measuring else {}),
                var_names=datasets['train'].target_names(),
                op_before_running=None,
                save_last_model=True,
                val_track_metric=model_class.VAL_TRACK_METRIC,
                **model_args)