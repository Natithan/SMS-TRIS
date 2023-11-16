"""
Run file for training CITRIS-NF on a pretrained autoencoder architecture.
"""
import jsonargparse
from experiments.domain_adapt import add_oodlog_callback_kwargs

from models.ae import Autoencoder
from pretorch_util import assign_visible_gpus
assign_visible_gpus(free_mem_based=True)
import os
import torch
import torch.utils.data as data
import sys
from myutil import add_id_ood_common_args, add_id_common_args, add_flow_args, get_cfd_indicator

sys.path.append('../')
from models.citris_nf import CITRISNF
from models.icitris_nf import iCITRISNF
from experiments.utils import train_model, load_datasets, get_default_parser, print_params, maybe_minify_args, update_aeckpt_arg


def main():
    parser = jsonargparse.ArgumentParser(default_config_files=['cfgs/CITRISNF_og.yml'])
    parser = get_default_parser(parser)
    add_id_common_args(parser)
    parser = add_custom_args(parser)
    add_id_ood_common_args(parser)
    args = parser.parse_args()

    args = update_train_nf_args(args)
    model_args = vars(args)
    datasets, data_loaders, data_name = load_datasets(args)
    assign_model_args(args, data_loaders, datasets, model_args)
    model_class = get_model_class(args)
    logger_name = get_logger_name(args, data_name, model_args)
    print_params(logger_name, model_args)
    check_val_every_n_epoch = get_val_check_freq(args, model_args)
    callback_kwargs = get_callback_kwargs(args, datasets, model_class)

    train_model(model_class=model_class,
                train_loader=data_loaders['train'],
                # val_loaders_dict ={
                #                       'triplet': data_loaders['val_triplet'] if not args.no_alignment_measuring else None,
                #                       'fp': data_loaders['val_fp']} |
                #                   ({} if args.skip_cma else {'cma_fp': data_loaders['val_cma_fp']}),
                val_loaders_dict={
                    'fp': data_loaders['val_fp']} |
                                    ({} if args.skip_cma else {'cma_fp': data_loaders['val_cma_fp']}) |
                                    ({} if args.no_alignment_measuring else {'triplet': data_loaders['val_triplet']}),

                # test_loaders_dict={
                #                       'triplet': data_loaders['test_triplet'] if not args.no_alignment_measuring else None,
                #                       'fp': data_loaders['test_fp']}
                #                   | ({} if args.skip_cma else {'cma_fp': data_loaders['test_cma_fp']}),
                test_loaders_dict={
                    'fp': data_loaders['test_fp']}
                                    | ({} if args.skip_cma else {'cma_fp': data_loaders['test_cma_fp']})
                                    | ({} if args.no_alignment_measuring else {'triplet': data_loaders['test_triplet']}),
                logger_name=logger_name,
                check_val_every_n_epoch=check_val_every_n_epoch,
                progress_bar_refresh_rate=0 if args.cluster else 1,
                callback_kwargs=callback_kwargs,
                var_names=datasets['train'].target_names(),
                op_before_running=lambda model: encode_datasets(model, list(datasets.values())),
                save_last_model=True,
                cluster_logging=args.cluster,
                # val_track_metric='val_comb_loss' if not args.use_baseline_correlation_metrics else 'val_comb_loss_grouped_latents',
                val_track_metric=model_class.DEFAULT_VAL_TRACK_METRIC,
                **model_args)


def get_callback_kwargs(args, datasets, model_class):
    callback_kwargs = {'dataset': datasets['train'],
                       'add_enco_sparsification': args.enco_postprocessing,
                       'mini': args.mini,
                       'pt_args': args,
                       'use_baseline_correlation_metrics': args.use_baseline_correlation_metrics, }
    if not args.no_alignment_measuring:
        callback_kwargs |= {
            'correlation_dataset': datasets['val_corr'],
            'correlation_test_dataset': datasets['test_corr'],
        }
    callback_kwargs = add_oodlog_callback_kwargs(args, callback_kwargs, datasets, model_class)
    return callback_kwargs


def update_train_nf_args(args):
    maybe_minify_args(args)
    update_bcm_arg(args)
    update_aeckpt_arg(args)
    if args.freeze_flow is None:
        if args.flow_wid is None:
            args.freeze_flow = False
            print("Not freezing flow by default")
        else:
            args.freeze_flow = True
            print("Freezing flow by default")
    return args


def update_bcm_arg(args):
    if args.use_baseline_correlation_metrics is None:
        args.use_baseline_correlation_metrics = (
                args.fullview_baseline and args.no_init_masking and args.no_context_masking)
    if args.use_baseline_correlation_metrics:
        print('Using baseline correlation metrics')


def get_val_check_freq(args, model_args):
    check_val_every_n_epoch = model_args.pop('check_val_every_n_epoch')
    if check_val_every_n_epoch <= 0:
        check_val_every_n_epoch = 5 if not args.cluster else 10
    return check_val_every_n_epoch


def get_model_class(args):
    if args.model == 'CITRISNF':
        model_class = CITRISNF
    elif args.model == 'iCITRISNF':
        model_class = iCITRISNF
    else:
        assert False, f'Unknown model class \"{args.model}\"'
    return model_class


def assign_model_args(args, data_loaders, datasets, model_args):
    model_args['data_folder'] = [s for s in args.data_dir.split('/') if len(s) > 0][-1]
    model_args['width'] = datasets['train'].get_img_width()
    model_args['num_causal_vars'] = datasets['train'].num_vars()
    DataClass = datasets['train'].__class__
    model_args['DataClass'] = DataClass
    model_args['max_iters'] = args.max_epochs * len(data_loaders['train'])


def get_logger_name(args, data_name, model_args):
    if not args.cluster_log_folder:
        if args.mini:
            maybe_mini = 'mini_'
        else:
            maybe_mini = ''
        flow_indication_prestring = ''
        if args.skip_flow:
            flow_indication_prestring = "SF_"
        elif args.random_flow:
            flow_indication_prestring = "RF_"
        elif args.flow_wid is not None:
            flow_indication_prestring = f"CC_{args.flow_wid}_"

        cfd_indicator = get_cfd_indicator(args)
        logger_name = f'{maybe_mini}{flow_indication_prestring}{args.model.replace("CI", cfd_indicator)}_{args.num_latents}l_{args.num_causal_vars}b_{args.c_hid}hid_{data_name}'
    else:
        logger_name = 'Cluster'
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name
    return logger_name




def encode_datasets(model, datasets, data_folder=None):
    if data_folder is None:
        data_folder = model.hparams.data_folder
    if isinstance(datasets, data.Dataset):
        datasets = [datasets]
    if any([isinstance(d, dict) for d in datasets]):
        new_datasets = []
        for d in datasets:
            if isinstance(d, dict):
                new_datasets += list(d.values())
            else:
                new_datasets.append(d)
        datasets = new_datasets
    for dataset in datasets:
        autoencoder_folder = model.hparams.autoencoder_checkpoint.rsplit('/', 1)[0]
        encoding_folder = os.path.join(autoencoder_folder, 'encodings/')
        os.makedirs(encoding_folder, exist_ok=True)
        # TODO check if this works correctly with deconf datasets. I think it doesn't!
        encoding_filename = os.path.join(encoding_folder, f'{data_folder}_{dataset.split_name}{dataset.maybe_mini}.pt') # Nathan: added the maybe_mini
        if hasattr(model, 'autoencoder'):
            encoder = model.autoencoder.encoder
        else: # Case for separately training flow
            encoder = Autoencoder.load_from_checkpoint(model.hparams.autoencoder_checkpoint).encoder
        if not os.path.exists(encoding_filename):
            encodings = dataset.encode_dataset(encoder)
            torch.save(encodings, encoding_filename)
        else:
            dataset.load_encodings(encoding_filename)


def add_custom_args(parser):

    parser.add_argument('-c', '--config', help="configuration file *.yml. Can be overriden with direct args",
                        action=jsonargparse.ActionConfigFile)
    parser.add_argument('--model', type=str, default='iCITRISNF')
    parser.add_argument('--c_hid', type=int, default=64)
    add_flow_args(parser)
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--hidden_per_var', type=int, default=16)
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--classifier_lr', type=float, default=4e-3)
    parser.add_argument('--classifier_momentum', type=float, default=0.0)
    parser.add_argument('--classifier_gumbel_temperature', type=float, default=1.0)
    parser.add_argument('--classifier_use_normalization', action='store_true')
    parser.add_argument('--classifier_use_conditional_targets', action='store_true')
    parser.add_argument('--beta_t1', type=float, default=1.0)
    parser.add_argument('--beta_classifier', type=float, default=2.0)
    parser.add_argument('--beta_mi_estimator', type=float, default=2.0)
    parser.add_argument('--lambda_reg', type=float, default=0.0)
    parser.add_argument('--num_graph_samples', type=int, default=8)
    parser.add_argument('--autoregressive_prior', action='store_true')
    parser.add_argument('--cluster_log_folder', action="store_true")
    parser.add_argument('--lambda_sparse', type=float, default=0.05) # Nathan changed default from .1 to .05
    parser.add_argument('--mi_estimator_comparisons', type=int, default=1)
    parser.add_argument('--graph_learning_method', type=str, default="ENCO")
    parser.add_argument('--use_notears_regularizer', action="store_true")
    parser.add_argument('--include_og_imgs', type=bool, default=True)
    parser.add_argument('--skip_flow', action="store_true")
    parser.add_argument('--random_flow', action="store_true")
    parser.add_argument('--per_latent_full_silos', action="store_true")
    parser.add_argument('--train_ae', action="store_true")
    parser.add_argument('--cheat_cfd', action="store_true")
    parser.add_argument('--flow_wid', type=str, default=None,
                        help='Wandb id for flow-part of model for which the path will be inferred')
    parser.add_argument('--freeze_flow', type=bool, default=None)
    return parser


if __name__ == '__main__':
    main()