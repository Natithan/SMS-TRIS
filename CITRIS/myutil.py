# Nathan
import argparse

from constants import TC3DI_ROOT, CE_CKPT


def add_id_ood_common_args(parser, from_ce=False):
    parser.add_argument('--mini', action='store_true')
    parser.add_argument('--nowandb', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=512 if not from_ce else 128)

    parser.add_argument('-r','--wandb_resume_id', type=str,default=None) # resume wandb run
    parser.add_argument('--ckpt_path', type=str, default=None) # mutually exclusive with wandb_resume_id
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--wandb_id', type=str, default=None) # if you want to specify a (new) wandb id
    parser.add_argument('--img_callback_freq', type=int, default=50)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=50)
    parser.add_argument('--enco_postprocessing', action="store_true")
    # Argument to add wandb notes
    parser.add_argument('--notes', type=str, default='')
    # wandb name prefix
    parser.add_argument('--prename', type=str, default='')
    # wandb name suffix
    parser.add_argument('--postname', type=str, default='')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None)

    # args for deep coral
    parser.add_argument('--coral', action="store_true")
    parser.add_argument('--fsl_seed', type=int, default=0)
    parser.add_argument('--ood_data_dir', type=str, default = None, help='If not specified, will use default OOD dataset corresponding to the ID dataset')
    parser.add_argument('--beta_coral', type=float, default=1.0)
    # parser.add_argument('--coral_upweight', action="store_true")

    parser.add_argument('--notrain', action="store_true")
    parser.add_argument('--no_intvs', action="store_true") # Whether to work with data in which there are never interventions
    parser.add_argument('--anti_align', action="store_true")
    parser.add_argument('--beta_anti_align', type=float, default=1.0)

    parser.add_argument('--std_min', type=float, default=0.0)
    parser.add_argument('--no_alignment_measuring', action="store_true")
    parser.add_argument('--use_nonzero_mask', action="store_true")
    parser.add_argument('--deconf', action="store_true", help='Whether to train on decorrelated data')
    parser.add_argument('--use_mean_to_autoregress_str', type=str, default="True", help='Can be "both", "True", or "False"')
    parser.add_argument('--per_batch_logging', action="store_true", help="If true, log per-batch values")
    parser.add_argument('--log_all_train_values', action="store_true", help="Whether to log all elements of the loss dict, can be large")
    parser.add_argument('--skip_0ep_train_eval', action="store_true", help="Whether to skip 0-ep train eval")
    parser.add_argument('--num_ood_epochs', type=int, default=50)
    parser.add_argument('--do_ood_logging', action="store_true")
    parser.add_argument('-manep','--manual_epochs_to_save_at', type=int, nargs='*', default=None)




def add_id_common_args(parser):
    parser.add_argument('--data_dir', type=str, required=True, default=TC3DI_ROOT)
    parser.add_argument('--load_pretrained', action="store_true")
    parser.add_argument('--ewid', type=str, default=None, help='Can be a full path to a checkpoint, or a wandb id from which the path will be inferred')
    parser.add_argument('--train_on_decorr', action="store_true", help='Whether to train on decorrelated data') # DEPRECATED in favor of --deconf
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--fixed_logstd', action="store_true")
    parser.add_argument('--sample_mean', action="store_true", help="If true, we always sample the mean and don't take the predicted variance into account")
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--causal_encoder_checkpoint', type=str,
                        required=True, default=CE_CKPT)  # Needed for all id settings because correlation logging is intertwined with the causal encoder class atm
    parser.add_argument('--skip_cma', type=bool, default=None, help='Whether to skip cma logging')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--fp_log_freq', type=int, default=10)
    # ood_log_nshots do_ood_logging ood_log_freq
    # parser.add_argument('--ood_log_nshots', type=int, default=200) # allow multiple values instead of just one
    parser.add_argument('--ood_log_nsshots', type=int, nargs='+', default=[0,200]) # allow multiple values instead of just one. Do this in command line as eg --ood_log_nshots 0 200 500 (according to copilot :D)
    parser.add_argument('--ood_log_freq', type=int, default=10)
    parser.add_argument('--ood_train_val_split', type=float, default=0.8)
    parser.add_argument('--ood_lr', type=float, default=1e-3)
    # ood_unfreeze_subset: string, one of 'uf_all', 'uf_cheat_pnu', 'uf_pred_pnu'
    parser.add_argument('--ood_unfreeze_subset', type=str, default='uf_all')
    parser.add_argument('--fullview_baseline', action="store_true")
    parser.add_argument('--no_init_masking', action="store_true")
    parser.add_argument('--no_context_masking', action="store_true")
    parser.add_argument('--use_baseline_correlation_metrics', type=bool, default=None)
    parser.add_argument('--only_one_dshift', action="store_true")
    parser.add_argument('--debug_xmse_logging', action="store_true")
    parser.add_argument('--debug_ood_logging', action="store_true")
    parser.add_argument('--debug_epoch', type=int, default=500)
    parser.add_argument('--debug_wid', type=str)



class Namespace(argparse.Namespace):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_parent(cls, parent): # for purpose of being able to do args | Namespace(...)
        return cls(**parent.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def __or__(self, other):
        return Namespace(**{**self.__dict__, **other.__dict__})


def get_level_by_name(df, level_name):
    for l in df.index.levels:
        if l.name == level_name:
            return l
    raise ValueError(f"Level name {level_name} not found in df")


def namespace_to_arglist(args):
    args_as_list = []
    for k, v in vars(args).items():
        if isinstance(v, list):
            args_as_list += [f"--{k}"] + [f"{el}" for el in v]
        elif isinstance(v, bool):
            if v:
                args_as_list += [f"--{k}"]
        elif v is None:
            continue
        else:
            args_as_list += [f"--{k}", f"{v}"]
    return args_as_list


def add_flow_args(parser):
    parser.add_argument('--num_flows', type=int, default=4)
    parser.add_argument('--flow_act_fn', type=str, default='silu')
    parser.add_argument('--num_latents', type=int, default=32)
    parser.add_argument('--autoencoder_checkpoint', type=str, default='')


def get_cfd_indicator(args):
    if not args.fullview_baseline:
        cfd_indicator = "CI"
    else:
        if args.beta_classifier != 0:
            cfd_indicator = "FV"
        else:
            cfd_indicator = "St"
    return cfd_indicator
