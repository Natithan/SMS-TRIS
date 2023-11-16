"""
Run file to train iCITRIS-VAE, CITRIS-VAE, iVAE*, and SlowVAE.
"""
import jsonargparse

from experiments.domain_adapt import add_oodlog_callback_kwargs
from experiments.train_nf import get_callback_kwargs
from pretorch_util import assign_visible_gpus
assign_visible_gpus(free_mem_based=True)
import sys
from myutil import add_id_ood_common_args, add_id_common_args, get_cfd_indicator

sys.path.append('../')
from models.icitris_vae import iCITRISVAE
from models.citris_vae import CITRISVAE
from models.baselines import iVAE, SlowVAE
from experiments.utils import train_model, load_datasets, get_default_parser, print_params, maybe_minify_args


def main():
    parser = jsonargparse.ArgumentParser(default_config_files=['cfgs/CITRISVAE_og.yml'])
    parser = get_default_parser(parser)
    parser = add_custom_args(parser)
    add_id_common_args(parser)
    add_id_ood_common_args(parser)
    args = parser.parse_args()

    maybe_minify_args(args)
    model_args = vars(args)
    datasets, data_loaders, data_name = load_datasets(args)
    assign_model_args(args, data_loaders, datasets, model_args)
    model_name = model_args.pop('model')
    model_class = get_model_class(model_name)
    logger_name = get_logger_name(args, data_name, model_args, model_name)
    print_params(logger_name, model_args)
    check_val_every_n_epoch = get_val_check_freq(args, model_args)
    # callback_kwargs={'dataset': datasets['train'],
    #  'correlation_dataset': datasets['val_corr'],
    #  'correlation_test_dataset': datasets['test_corr'],
    #  'add_enco_sparsification': args.enco_postprocessing,
    #  'mini': args.mini}
    #
    # callback_kwargs = add_oodlog_callback_kwargs(args, callback_kwargs, datasets, model_class)
    callback_kwargs = get_callback_kwargs(args, datasets, model_class)

    train_model(model_class=model_class,
                train_loader=data_loaders['train'],
                val_loaders_dict={"triplet": data_loaders['val_triplet'], "fp": data_loaders['val_fp']},
                test_loaders_dict={'triplet': data_loaders['test_triplet']},
                logger_name=logger_name,
                check_val_every_n_epoch=check_val_every_n_epoch,
                progress_bar_refresh_rate=0 if args.cluster else 1,
                callback_kwargs=callback_kwargs,
                var_names=datasets['train'].target_names(),
                save_last_model=True,
                cluster_logging=args.cluster,
                # val_track_metric='val_comb_loss', # Nathan: changed from defaulting to val_loss, which in og implementation is val_triplet_loss (aka w/o r2/spearman)
                val_track_metric=model_class.DEFAULT_VAL_TRACK_METRIC,
                **model_args)


def get_val_check_freq(args, model_args):
    check_val_every_n_epoch = model_args.pop('check_val_every_n_epoch')
    if check_val_every_n_epoch <= 0:
        check_val_every_n_epoch = 2 if not args.cluster else 25
    return check_val_every_n_epoch


def get_model_class(model_name):
    if model_name == 'iCITRISVAE':
        model_class = iCITRISVAE
    elif model_name == 'CITRISVAE':
        model_class = CITRISVAE
    elif model_name == 'iVAE':
        model_class = iVAE
    elif model_name == 'SlowVAE':
        model_class = SlowVAE
    return model_class


def assign_model_args(args, data_loaders, datasets, model_args):
    model_args['data_folder'] = [s for s in args.data_dir.split('/') if len(s) > 0][-1]
    model_args['img_width'] = datasets['train'].get_img_width()
    model_args['num_causal_vars'] = datasets['train'].num_vars()
    DataClass = datasets['train'].__class__
    model_args['DataClass'] = DataClass
    model_args['max_iters'] = args.max_epochs * len(data_loaders['train'])
    if hasattr(datasets['train'], 'get_inp_channels'):
        model_args['c_in'] = datasets['train'].get_inp_channels()


def get_logger_name(args, data_name, model_args, model_name):
    if args.mini:
        maybe_mini = 'mini_'
    else:
        maybe_mini = ''
    cfd_indicator = get_cfd_indicator(args)
    logger_name = f'{maybe_mini}{model_name.replace("CI", cfd_indicator)}_{args.num_latents}l_{model_args["num_causal_vars"]}b_{args.c_hid}hid_{data_name}'
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name
    return logger_name


def add_custom_args(parser):

    parser.add_argument('-c', '--config', help="configuration file *.yml. Can be overriden with direct args",
                        action=jsonargparse.ActionConfigFile)
    parser.add_argument('--model', type=str, default='iCITRISVAE')
    parser.add_argument('--c_hid', type=int, default=32)
    parser.add_argument('--decoder_num_blocks', type=int, default=1)
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--num_latents', type=int, default=16)
    parser.add_argument('--classifier_lr', type=float, default=4e-3)
    parser.add_argument('--classifier_momentum', type=float, default=0.0)
    parser.add_argument('--classifier_gumbel_temperature', type=float, default=1.0)
    parser.add_argument('--classifier_use_normalization', action='store_true')
    parser.add_argument('--classifier_use_conditional_targets', action='store_true')
    parser.add_argument('--kld_warmup', type=int, default=0)
    parser.add_argument('--beta_t1', type=float, default=1.0)
    parser.add_argument('--beta_classifier', type=float, default=2.0)
    parser.add_argument('--beta_mi_estimator', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lambda_reg', type=float, default=0.0)
    parser.add_argument('--autoregressive_prior', action='store_true')
    parser.add_argument('--lambda_sparse', type=float, default=0.02)
    parser.add_argument('--mi_estimator_comparisons', type=int, default=1)
    parser.add_argument('--graph_learning_method', type=str, default="ENCO")
    parser.add_argument('--use_flow_prior', action='store_true')
    parser.add_argument('--use_notears_regularizer', action="store_true")
    parser.add_argument('--include_og_imgs', type=bool, default=True)
    return parser


if __name__ == '__main__':
    main()