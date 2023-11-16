# region imports
import json
import os
import re
import sys
import time
import warnings
import copy
import socket
from experiments.ood_utils import reset_model_and_optimizer, add_maybe_override_args, get_loader, MySubset, \
    FullID_FewShotOOD_Dataset, reset_model, reset_optimizer
from models.citris_vae import CITRISVAE
from myutil import add_id_ood_common_args, get_level_by_name, Namespace

from wandb_extensions import NonVersioningWandbImage

warnings.filterwarnings("ignore")
# warnings.filterwarnings(action="once")
# warnings.filterwarnings("ignore", message="Failed to load image Python extension")
# warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame")
from pretorch_util import assign_visible_gpus

assign_visible_gpus(free_mem_based=True)
import torch
from util import *
from experiments.datasets import Causal3DDataset, InterventionalPongDataset
from models.citris_nf import CITRISNF
from models.shared.transition_prior import OldMixedTrueLatentPrior
from models.mixed_factor_predictor import MixedFactorPredictor
from os.path import join as jn

from constants import CITRIS_ROOT, CITRIS_CKPT_ROOT, P2_ROOT, TC3DI_ROOT, PONG_ROOT, PONG_OOD_ROOT, TC3DI_OOD_ROOT
import pandas as pd
from tqdm import tqdm
import jsonargparse
import wandb
import seaborn as sns
import matplotlib.pyplot as plt

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
# endregion

gauth = GoogleAuth(settings_file=jn(P2_ROOT, 'gdrive_settings.yaml'))
drive = GoogleDrive(gauth)

DEFAULT_METRICS = ['nll', 'mse', 'mae', 'logstd', 'normed_dist','minlogstd','maxlogstd']
DEFAULT_FEATURE_LEVELS = ['xt1', 'yt1', 'zt1', 'zphit1', 'ct1']

def minify_args(args):
    print("Setting epochs to 1 because mini flag is set")
    args.num_epochs = 1 if not (args.num_epochs < 0) else -args.num_epochs
    n = 2
    # args.ood_factors_list = args.ood_factors_list[:n]
    # args.uf_groups = (args.uf_groups[:n] +
    #                         args.uf_groups[len(args.data_class.FACTORS) - 1:len(args.data_class.FACTORS) - 1 + n] +
    #                         args.uf_groups[len(args.data_class.FACTORS) - 1 + len(args.data_class.COARSE_FACTORS) - 1:len(args.data_class.FACTORS) + len(args.data_class.COARSE_FACTORS) - 1 + n])
    args.loss_factors = args.loss_factors[:n]
    if args.test_batch_size >= 0: print("Setting test batch size to 13 because mini flag is set")
    args.test_batch_size = 13 if args.test_batch_size >= 0 else -args.test_batch_size
    if args.batch_size >= 0: print("Setting batch size to 13 because mini flag is set")
    args.batch_size = 13 if args.batch_size >= 0 else -args.batch_size
    # args.coral_batch_size = 13 if args.coral_batch_size >= 0 else -args.coral_batch_size
    # if args.coral_batch_size >=0: print("Setting coral batch size to 13 because mini flag is set")

    return args

def get_data_class(data_name):
    if (data_name == 'causal3d') or (data_name == 'shapes'):
        return Causal3DDataset
    elif data_name == 'pong':
        return InterventionalPongDataset
    else:
        raise NotImplementedError

def get_ood_root_dir_for_data_class(data_class):
    if data_class == Causal3DDataset:
        # return jn(CITRIS_ROOT,
        #               "data_generation/temporal_causal3dident/1000_points")
        # 100k points
        # return jn(CITRIS_ROOT,"data_generation/temporal_causal3dident/100000_points")
        return TC3DI_OOD_ROOT
    elif data_class == InterventionalPongDataset:
        # return jn(CITRIS_ROOT, "data_generation/ood_pong/10000_train_points/")
        return PONG_OOD_ROOT
    else:
        raise NotImplementedError

def get_per_factor_grad_norm(model):
    c_hid = model.hparams.c_hid
    c_out = model.hparams.c_out
    for name, param in model.named_parameters():
        # if name.startswith('context_layer') or name.startswith('target_layer'):
        #     unflattened_grad = param.grad.unflatten(0, (-1, c_hid))
        #     if name.startswith('context_layer') and name.endswith('weight'):
        #         per_parent_grad_norm = unflattened_grad.norm(dim=1)
        #         num_parents_per_out_factor = model.context_layer.mask.unflatten(0,(-1,c_hid))[:,0].sum(-1)
        #         return per_parent_grad_norm.sum(1) / num_parents_per_out_factor
        #     else:
        #         return unflattened_grad.norm(dim=1).mean(1)
        # elif name.startswith('net'):
        return param.grad.mean(dim=list(range(len(param.shape)))[1:])  # all but first dim, which is per-output-factor
        # else:
        #     raise NotImplementedError(f"Unknown parameter name {name}")


# def get_mean_per_factor_grad_norm(model, dataloader, distribution, optimizer):
#     per_factor_grad_norm_list = []
#     for batch in dataloader:
#         raise NotImplementedError("TODO update when I want to use this method again, this now returns a dict")
#         train_loss = get_train_loss(batch, model)
#         optimizer.zero_grad()
#         train_loss.backward()
#         per_factor_grad_norm = get_per_factor_grad_norm(model)
#         per_factor_grad_norm_list.append(per_factor_grad_norm)
#     mean_per_factor_grad_norm = torch.stack(per_factor_grad_norm_list).mean(0)
#     return mean_per_factor_grad_norm

def log_subdf_as_pivot(subdf, model_id, name,pivot_df_dict, args, index=('distribution','n_shots'),columns='loss_factor', metrics=('nll','mse','mae','logstd')):
    for metric in metrics:
        name_postfix = "_" + str(metric)
        full_name = name + name_postfix
        subdf_to_pivot = subdf[subdf.metric == metric]
        pivoted_df = subdf_to_pivot.pivot(index=index,
                                               columns=columns,
                                               values='value')
        pivot_df_dict[full_name] = pivoted_df
        if len(args.wandb_ids) > 1:
            title = f'{model_id}-{full_name}'
        else:
            title = full_name
        # check if not all NaN
        if not pivoted_df.isnull().values.all():
            log_pivot(df=pivoted_df, title=title, args=args)


def main():
    # region parse args
    parser = jsonargparse.ArgumentParser(default_config_files=['cfgs/DA.yml'])
    add_specific_args(parser)

    add_id_ood_common_args(parser)
    args = parser.parse_args()
    args = update_da_args(args)
    # endregion
    set_pandas_full_print()

    if args.nowandb:  # For when debugging
        os.environ['WANDB_MODE'] = 'disabled'
    loss_df_last, travel_df_last, model_id2ckpt_path, model_id2model_class, model_id2is_nan = set_up_dfs(args)
    loss_df_best_val, travel_df_best_val = loss_df_last.copy(), (travel_df_last.copy() if travel_df_last is not None else None)
    set_run_name(args, model_id2ckpt_path)
    loss_df_csv_dir = get_loss_df_csv_dir(args.run_name)
    setup_wandb(args, loss_df_csv_dir)
    df_root_dir = (wandb.run.dir.split('wandb/')[
                                      0] if not args.nowandb else wandb.run.dir + "/")
    # pivot_dfs = {}
    pivot_dfs_last, pivot_dfs_best_val = {}, {}
    # gradient_df = pd.DataFrame(index=get_level_by_name(df,'distribution'),columns=) # TODO: maybe add gradient df, if considered complementary to per-factor-loss df
    # id_pretrained_models = {mid: None for mid in get_level_by_name(loss_df_last, 'model')}
    id_pretrained_models = {mid: {'last': None, 'best_val': None} for mid in get_level_by_name(loss_df_last, 'model')}
    # distr2logstd_preds = {}
    distr2logstd_preds_last, distr2logstd_preds_best_val = {}, {}

    # Get id train loader to allow comparing losses to find PNU
    id_train_loader = get_loader(args, args.data_class, args.iid_data_dir, 'train')

    for distribution, group in loss_df_last.groupby(level=0):
        for n_shots, group in group.groupby(level=1):
            if args.coral and n_shots == 1:
                print(f"Skipping {distribution} with {n_shots} shots for coral, because the covariance matrix need at least 2 OOD samples")
                continue
            loaders = load_few_shot_datasets(args, distribution, n_shots)
            for model_id, group in group.groupby(level=2):
                if model_id2is_nan[model_id]:
                    print(f"Skipping {model_id} because it is nan")

                    continue
                if distribution != 'ID':
                    # pivot_dfs[model_id] = {}
                    pivot_dfs_last[model_id], pivot_dfs_best_val[model_id] = {}, {}
                ckpt_path = model_id2ckpt_path[model_id]
                model_class = model_id2model_class[model_id]
                ood_shifts = None if distribution == 'ID' else distribution.split('OOD_')[1].split('__')
                model, optimizer, scheduler_dict = reset_model_and_optimizer(ckpt_path, model_class, args,
                                                                             # uf_prior_only=args.prior_only,
                                                                             unfreeze_subset=None,
                                                                             ood_shifts=ood_shifts, loaders=loaders, id_train_loader=id_train_loader)
                model_last, model_best_val = model, model
                if args.coral:
                    model_best_val, model_last = get_inThisScript_pretrained_models(args, ckpt_path, distribution,
                                                                                    id_pretrained_models, id_train_loader, loaders,
                                                                                    model_class, model_id, n_shots, ood_shifts)
                for unfreeze_subset, group in group.groupby(level=3):
                        # if n_shots == 0 and unfreeze_factor == 'uf_all' and distribution != 'ID':
                        #     get_mean_per_factor_grad_norm(model, loaders['train'], distribution, optimizer)

                        # If args.coral, the model starts from random and interleaves ID and available shots of OOD data.
                        # For 0-shot ID and 0-shot-OOD evaluation, training is the same: only on ID data, so this training needs to happen only once, and can then be reused for all 0-shot evaluations.
                        # if args.coral and ((n_shots > 0) or id_pretrained_models[model_id]['last'] is None):


                        # if n_shots > 0 or (args.coral and id_pretrained_models[model_id]['last'] is None):
                        if n_shots > 0:
                            # model, optimizer, scheduler_dict = reset_model_and_optimizer(ckpt_path, model_class, args,
                            #                                                              # uf_prior_only=args.prior_only,
                            #                                                              unfreeze_subset=unfreeze_subset,
                            #                                                              ood_shifts=ood_shifts, loaders=loaders, id_train_loader=id_train_loader)
                            if args.coral:
                                model = copy.deepcopy(model_best_val) # Use the best model from the pretraining phase
                            else:
                                model = reset_model(args, ckpt_path, model_class, ood_shifts, id_train_loader)
                            optimizer, scheduler_dict = reset_optimizer(args, loaders, model, unfreeze_subset=unfreeze_subset)
                            dct = train_da(args, model, n_shots, optimizer, scheduler_dict, loaders['train'], loaders['val'], unfreeze_setting=unfreeze_subset, distribution=distribution)
                            # model_last, avg_param_travel_last = dct['last']['model'], dct['last']['avg_param_travel']
                            # model_best_val, avg_param_travel_best_eval = dct['best_val']['model'], dct['best_val']['avg_param_travel']
                            model_last, model_best_val = dct['last']['model'], dct['best_val']['model']
                            if not args.coral:
                                update_travel_df(dct, distribution, model_id, n_shots, travel_df_best_val,
                                                 travel_df_last, unfreeze_subset)
                            # if n_shots == 0: # this can only happen if args.coral. Use last model in that case
                            #     id_pretrained_models[model_id]['last'] = model_last
                            #     id_pretrained_models[model_id]['best_val'] = model_best_val
                        # elif args.coral: # case n_shots == 0 and model already trained on ID
                        #     assert id_pretrained_models[model_id]['last'] is not None
                        #     model_last = id_pretrained_models[model_id]['last']
                        #     model_best_val = id_pretrained_models[model_id]['best_val']
                        print(f'Evaluating {model_id} on {distribution} with {n_shots} shots and {unfreeze_subset}')
                        # loss_dict = ttest_da(args, model, loaders['test'],distribution)
                        # fill_loss_df(loss_df_last, loss_dict, distribution, model_id, n_shots, unfreeze_factor,args)
                        # loss_df_last.reset_index().to_csv(df_root_dir + 'full_loss_df.csv')
                        # # Log pivot tables
                        # if distribution != 'ID':
                        #     log_pivot_dfs(args, loss_df_last, model_id, pivot_dfs, unfreeze_factor)
                        # if args.log_logstd_hists:
                        #     log_logstd_preds(distr2logstd_preds, distribution, loss_dict, model_id, n_shots,
                        #                      unfreeze_factor)
                        for mdl, loss_df, pivot_dfs, distr2logstd_preds, mode in zip([model_last, model_best_val], [loss_df_last, loss_df_best_val], [pivot_dfs_last, pivot_dfs_best_val], [distr2logstd_preds_last, distr2logstd_preds_best_val], ['last', 'best_val']):
                            loss_dict = ttest_da(args, mdl, loaders['test'], distribution)
                            fill_loss_df(loss_df, loss_dict, distribution, model_id, n_shots, unfreeze_subset, args)
                            loss_df.reset_index().to_csv(df_root_dir + f'full_loss_df_{mode}.csv')
                            # Log pivot tables
                            if distribution != 'ID':
                                log_pivot_dfs(args, loss_df, model_id, pivot_dfs, unfreeze_subset, mode=mode)
                            if args.log_logstd_hists:
                                log_logstd_preds(distr2logstd_preds, distribution, loss_dict, model_id, n_shots,
                                                 unfreeze_subset, mode=mode)
                        print(f'Finished evaluating {model_id} on {distribution} with {n_shots} shots and {unfreeze_subset}')
    full_loss_df = loss_df_last.reset_index()
    wandb.summary['full_loss_df'] = wandb.Table(dataframe=full_loss_df)  # Maybe superfluous
    if args.log_0sh_worsening:
        abs_diff_df = get_diff_subdf(full_loss_df, absolute=True)
        # df with 1 row, and columns: c_match, c_nonmatch, c_mean. Log to wandb as bar plot
        wandb.log({'id_to_0shood_worsening_absolute_barplot': wandb.plot.bar(wandb.Table(data=abs_diff_df.melt().rename(columns={'loss_factor': 'c_aggregate'})), 'c_aggregate', 'value', title='Absolute cnd worsening from ID to 0sh-OOD')})


def get_inThisScript_pretrained_models(args, ckpt_path, distribution, id_pretrained_models, id_train_loader,
                                       loaders, model_class, model_id, n_shots, ood_shifts):
    id_model_ready = id_pretrained_models[model_id]['last'] is not None
    zero_sh = n_shots == 0
    share_for_all_sh = args.beta_coral == 0
    if share_for_all_sh:
        model_ready_for_current_sh = id_model_ready
    else:
        model_ready_for_current_sh = zero_sh and id_model_ready
    # if (not zero_sh or not model_ready_for_0sh):
    if not model_ready_for_current_sh:
        model, optimizer, scheduler_dict = reset_model_and_optimizer(ckpt_path, model_class, args,
                                                                     # uf_prior_only=args.prior_only,
                                                                     unfreeze_subset='uf_all_enc',
                                                                     ood_shifts=ood_shifts,
                                                                     loaders=loaders,
                                                                     id_train_loader=id_train_loader)
        pre_dct = train_da(args, model, n_shots=n_shots, optimizer=optimizer,
                           scheduler_dict=scheduler_dict, train_loader=loaders['pretrain'],
                           val_loader=loaders['preval'], unfreeze_setting='uf_all_enc',
                           distribution=distribution, pretrain_phase=True)
        model_best_val = pre_dct['best_val']['model']
        model_last = pre_dct['last']['model']
        if zero_sh or share_for_all_sh:
            id_pretrained_models[model_id]['last'] = model_last
            id_pretrained_models[model_id]['best_val'] = model_best_val
    else:
        model_best_val = id_pretrained_models[model_id]['best_val']
        model_last = id_pretrained_models[model_id]['last']
    return model_best_val, model_last


def update_travel_df(dct, distribution, model_id, n_shots, travel_df_best_val, travel_df_last, unfreeze_subset):
    avg_param_travel_last, avg_param_travel_best_eval = dct['last']['avg_param_travel'], dct['best_val'][
        'avg_param_travel']
    # for k, v in avg_param_travel.items():
    #     travel_df_last.loc[(distribution, n_shots, model_id, unfreeze_factor, k), 'value'] = v
    # travel_df_last.reset_index().to_csv(df_root_dir + 'travel_df.csv')
    for avg_param_travel, travel_df in zip([avg_param_travel_last, avg_param_travel_best_eval],
                                           [travel_df_last, travel_df_best_val]):
        for k, v in avg_param_travel.items():
            travel_df.loc[(distribution, n_shots, model_id, unfreeze_subset, k), 'value'] = v


def setup_wandb(args, loss_df_csv_dir):
    try:
        wandb.init(project='p2_citris', entity='liir-kuleuven', dir=loss_df_csv_dir, name=args.run_name,
                   notes=args.notes,
                   tags=args.wandb_tags, )
    except wandb.errors.UsageError:
        # try with settings=wandb.Settings(start_method='fork'
        wandb.init(project='p2_citris', entity='liir-kuleuven', dir=loss_df_csv_dir, name=args.run_name,
                   notes=args.notes,
                   tags=args.wandb_tags, settings=wandb.Settings(start_method='fork'))
    wandb.config.update(args)
    wandb.config['cli_args'] = " ".join(sys.argv[1:])


def set_run_name(args, model_id2ckpt_path):
    if args.run_name == '':
        args.run_name = args.prename + \
                        ('mini_' if args.mini else '') + \
                        'DA__' + \
                        (f'{args.pt_epoch}_' if args.pt_epoch is not None else '') + \
                        '_'.join(model_id2ckpt_path) + \
                        args.postname


def log_logstd_preds(distr2logstd_preds, distribution, loss_dict, model_id, n_shots, unfreeze_factor, mode='last'):
    if (unfreeze_factor == 'uf_all') and (n_shots == 0):
        ls_og = loss_dict['prior_logstd_per_hyp']
        ls_psi = loss_dict['softpsi_adjusted_prior_logstd_per_hyp']
        distr2logstd_preds[model_id + '_' + distribution] = {'ls_og': ls_og, 'ls_psi': ls_psi}
        if distribution != 'ID':
            N, Z, Cplus1 = ls_og.shape
            n_bins = 100
            fig_og, axs_og = plt.subplots(Cplus1, Z, figsize=(Z * 5, Cplus1 * 5))
            fig_psi, axs_psi = plt.subplots(Cplus1, Z, figsize=(Z * 5, Cplus1 * 5))
            for c in range(Cplus1):
                for z in range(Z):
                    print(f"Plotting {c} {z}")
                    for k, v in distr2logstd_preds.items():
                        args_og = {"x": v['ls_og'][:, z, c],
                                   # # B(L-1)_S0_Z_(C+1) -> B(L-1)S0_Z_(C+1)
                                   "bins": n_bins,
                                   "color": 'blue' if 'ID' in k else 'red',
                                   "label": k}
                        _ = axs_og[c, z].hist(**args_og, alpha=0.5)
                        args_psi = {"x": v['ls_psi'][:, z, c],
                                    # # B(L-1)_S0_Z_(C+1) -> B(L-1)S0_Z_(C+1)
                                    "bins": n_bins,
                                    "color": 'blue' if 'ID' in k else 'red',
                                    "label": k}
                        _ = axs_psi[c, z].hist(**args_psi, alpha=0.5)
                    for axs in [axs_og, axs_psi]:
                        axs[c, z].set_title(f"{c} {z}")
                        axs[c, z].legend()
                        axs[c, z].set_yscale('log')
            fig_og.savefig(f'tmp_logstd_preds_{model_id}_{distribution}_og.png')
            fig_psi.savefig(f'tmp_logstd_preds_{model_id}_{distribution}_psi.png')


def log_pivot_dfs(args, loss_df, model_id, pivot_dfs, unfreeze_factor, mode='last'):
    uf_subdf = loss_df.loc[:, :, model_id, unfreeze_factor].reset_index()
    log_subdf_as_pivot(uf_subdf, model_id, name=f'{unfreeze_factor}-{mode}', pivot_df_dict=pivot_dfs[model_id], args=args,
                       columns='loss_factor', metrics=uf_subdf.metric.unique())
    # for loss_mean in [f for f in get_level_by_name(loss_df,'loss_factor') if f.endswith('_mean')]:
    #     lf_subdf = loss_df.loc[:, :, model_id, :, loss_mean].reset_index()
    #     log_subdf_as_pivot(lf_subdf, model_id, name=loss_mean, pivot_df_dict=pivot_dfs[model_id],args=args,columns='unfreeze_factor',metrics=lf_subdf.metric.unique())
    # if any(uf not in ('uf_all') for uf in get_level_by_name(loss_df, 'unfreeze_factor')):
    #     uf_matching_subdf = loss_df.loc[:, :, model_id].reset_index()
    #
    #     # Filter so that distribution is ID or distribution factor after 'OOD_' matches unfreeze factor after 'uf_'
    #     def match_distr_and_uf(distr, uf):
    #         if distr in ['ID', 'OOD_none']:
    #             return uf == 'uf_all'
    #         else:
    #             distr_shifts = distr.split('OOD_')[1].split('__')
    #             possible_uf_shifts = (distr_shifts, [args.data_class.SHORT_FINE2COARSE[f] for f in distr_shifts])
    #             return uf.split('uf_')[1].split('__') in possible_uf_shifts
    #
    #     uf_matching_subdf = uf_matching_subdf[
    #         uf_matching_subdf.apply(lambda row: match_distr_and_uf(row.distribution, row.unfreeze_factor), axis=1)]
    #     log_subdf_as_pivot(uf_matching_subdf, model_id, name=f'uf_matching-{mode}', pivot_df_dict=pivot_dfs[model_id],
    #                        args=args, columns='loss_factor', metrics=uf_matching_subdf.metric.unique())
        # for value in 'loss_value', 'mae_value':
        #     name_postfix = '' if value == 'loss_value' else '_mae'
        #     uf_lf_subdf = loss_df.loc[:, :, model_id].reset_index()
        #     uf_lf_subdf = uf_lf_subdf[
        #         ((uf_lf_subdf.distribution == 'ID') & (uf_lf_subdf.unfreeze_factor == 'uf_all')) |
        #         (uf_lf_subdf.distribution.str.startswith('OOD_') &
        #          (uf_lf_subdf.unfreeze_factor.str.endswith('uf_all') |  # eg uf_all loss_x
        #           (uf_lf_subdf.unfreeze_factor == 'uf_' +
        #            uf_lf_subdf.loss_factor.str.split('loss_').str[1]) |  # eg uf_x loss_x
        #           (uf_lf_subdf.unfreeze_factor == 'uf_' +
        #            uf_lf_subdf.loss_factor.str.split('loss_').str[1].apply(
        #                get_uf_ext_for_loss_factor))
        #           # eg uf_x loss_0
        #           )
        #          )
        #         ]
        #     name_postfix = '' if value == 'loss_value' else '_mae'
        #     name = 'uf_lf' + name_postfix
        #     pivot_dfs[model_id][name] = uf_lf_subdf.pivot(index=['distribution', 'n_shots'],
        #                                                      columns=['loss_factor', 'unfreeze_factor'],
        #                                                      values=value)
        #     log_pivot(df=pivot_dfs[model_id][name], title=f'{model_class.__name__}-{name}', args=args)
        # uf_lf_subdf = loss_df.loc[:, :, model_id].reset_index()

        # uf_lf_subdf = uf_lf_subdf[
        #     ((uf_lf_subdf.distribution == 'ID') & (uf_lf_subdf.unfreeze_factor == 'uf_all')) |
        #     (uf_lf_subdf.distribution.str.startswith('OOD_') &
        #         (uf_lf_subdf.unfreeze_factor.str.endswith('uf_all') |  # eg uf_all loss_x
        #             (uf_lf_subdf.unfreeze_factor == 'uf_' +
        #             uf_lf_subdf.loss_factor.str.split('loss_').str[1]) |  # eg uf_x loss_x
        #             (uf_lf_subdf.unfreeze_factor == 'uf_' +
        #             uf_lf_subdf.loss_factor.str.split('loss_').str[1].apply(
        #                 get_uf_ext_for_loss_factor))
        #             # eg uf_x loss_0
        #         )
        #     )
        # ]

        # uf_lf_subdf = uf_lf_subdf[
        #     uf_lf_subdf.apply(
        #         lambda row: match_distr_and_lf_and_uf(args, row.distribution, row.unfreeze_factor, row.loss_factor), axis=1)]
        # log_subdf_as_pivot(uf_lf_subdf, model_id, name='uf_lf'-{mode}, pivot_df_dict=pivot_dfs[model_id], args=args,
        #                    columns=['loss_factor', 'unfreeze_factor'], metrics=uf_lf_subdf.metric.unique())
        # pivot_dfs[model_id]['uf_lf_mae'] = uf_lf_subdf.pivot(index=['distribution', 'n_shots'],
        #                                                  columns=['loss_factor', 'unfreeze_factor'],
        #                                                  values='mae_value')
        # log_pivot(df=pivot_dfs[model_id]['uf_lf_mae'], title=f'{model_class.__name__}-uf_lf_mae', args=args)


def match_distr_and_lf_and_uf(args, distr, uf, lf):
    if uf == 'uf_all':
        return True
    if distr == 'ID':
        return uf == 'uf_all'
    lf_shift = lf.split('_')[-1]
    uf_shifts = uf.split('uf_')[1].split('__')
    if (lf_shift in uf_shifts) or ((lf_shift in args.data_class.SHORT_FINE2COARSE) and (
            args.data_class.SHORT_FINE2COARSE[lf_shift] in uf_shifts)):
        return True
    return False


def add_specific_args(parser):
    parser.add_argument('-c', '--config', help="configuration file *.yml. Can be overriden with direct args",
                        action=jsonargparse.ActionConfigFile)
    parser.add_argument('-n', '--ns_shots', nargs="*", type=int)
    parser.add_argument('--ckpt_paths', nargs="+", type=str)
    parser.add_argument('-w', '--wandb_ids', nargs="+", type=str)
    parser.add_argument('-o', '--ood_factors_list', nargs="+", action='append', default=None)
    parser.add_argument('-u', '--uf_groups', nargs="*", action='append', default=None)
    parser.add_argument('-l', '--loss_factors', nargs="+", default=None)
    parser.add_argument('--metrics', nargs="+", default=DEFAULT_METRICS)
    parser.add_argument('--feature_levels', nargs="+", default=DEFAULT_FEATURE_LEVELS)
    parser.add_argument('--iid_data_dir', type=str,
                        default=None)
    # parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--coral_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs to few-shot train for")
    parser.add_argument('--prior_only', type=bool, default=False, help="Whether to only train the prior")
    parser.add_argument('--piecewise_unfreeze', type=bool, default=False)
    parser.add_argument('--skip_ID', action="store_true", help="Whether to skip ID evaluation")
    parser.add_argument('--skip_0shot', action="store_true", help="Whether to skip 0-shot evaluation")
    # parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--skip_plt', action="store_true", help="Whether to skip time-consuming logging of heatmaps")
    parser.add_argument('--log_logstd_hists', action="store_true", help="Whether to log histograms of logstds")
    parser.add_argument('--expl_ckpt_paths', nargs="+", type=str, default=None)
    parser.add_argument('--expl_model_classes', nargs="+", type=str, default=None)
    parser.add_argument('--expl_data_name', type=str, default=None)
    parser.add_argument('--expl_iid_data_dir', type=str, default=None)
    parser.add_argument('--reset_matching', action="store_true",
                        help="If true, reset parameters that match the distribution shift (assuming there are such parameters, eg for PMFP) to random")
    parser.add_argument('--sparse_update', action="store_true",
                        help="If true, only allow updating some sparse subset of parameters, ideally those that match the distribution shift") # TODO remove in favor of using unfreeze_factor
    parser.add_argument('--skip_uf_all', action="store_true",
                        help="If true, don't evaluate uf_all")
    parser.add_argument('--train_val_split', type=float, default=0.8)
    parser.add_argument('--fixed_logstd', type=bool, default=None, help="If not set, preserve setting of loaded model. If set, override setting of loaded model")
    parser.add_argument('--sample_mean', type=bool, default=None, help="If not set, preserve setting of loaded model. If set, override setting of loaded model")
    parser.add_argument('--log_0sh_worsening', action="store_true", help="If true, log the worsening of 0-shot performance compared to ID")
    parser.add_argument('--use_maxcorr_psi', type=bool, default=None, help="If not set, use default setting of loaded model. If set, override setting of loaded model")
    parser.add_argument('--log_heatmaps', action="store_true", help="If true, log heatmaps of cnd performance")
    parser.add_argument('--pt_epoch', type=int, default=None, help="If set, load model from this epoch")


def fill_loss_df(loss_df, loss_dict, distribution, model_id, n_shots, unfreeze_factor, args):
    subdf = loss_df.loc[distribution, n_shots, model_id, unfreeze_factor]

    for lf, metric in subdf.index:
        if any([lf.endswith(x) for x in ['mean', 'sum']]):
            dict_key = f'{metric}_{lf}'
            suffix = '_sar' if args.use_mean_to_autoregress_str == 'False' else '_mar'
            # if dict_key in loss_dict:
            if dict_key in loss_dict or dict_key + suffix in loss_dict:
                subdf.loc[lf, metric] = loss_dict[dict_key + (suffix if dict_key + suffix in loss_dict else '')]
        else:
            dict_key = f'{metric}_per_{"_".join(lf.split("_")[:-1])}'
            if lf.split('_')[-1].isdigit(): # c_mx_0, c_mx_1, ...
                idx = int(lf.split('_')[-1])
            elif re.match(f'c\_({"|".join(args.data_class.SHORT_FACTORS)})', lf): # c_x, c_y, c_rs, ...
                idx = args.data_class.SHORT_FACTORS.index(lf.split('_')[-1])
            elif re.match(f'zpsi\_({"|".join(args.data_class.SHORT_INTERVENED_FACTORS_AND_TRASH)})', lf): # zpsi_p, zpsi_r, zpsi_rs, .... + "t" for the trash dimension
                idx = (args.data_class.SHORT_INTERVENED_FACTORS_AND_TRASH).index(lf.split('_')[-1])
            else:
                raise ValueError(f'Unknown loss factor {lf}')
            # if dict_key in loss_dict:
            if dict_key in loss_dict or dict_key + suffix in loss_dict:
                subdf.loc[lf, metric] = loss_dict[dict_key + (suffix if dict_key + suffix in loss_dict else '')][idx]


def update_da_args(args):
    # if args.coral:
    #     args.batch_size = args.coral_batch_size
    #     print(f"Overriding uf_groups from {args.uf_groups} to ['all_enc'] for coral")
    #     args.uf_groups = [['all_enc']]
    #     args.skip_uf_all = True
    # if args.piecewise_unfreeze:
    #     args.prior_only = True
    args_to_ckpt_paths_and_model_classes_and_data_name(args)
    args.iid_data_dir = get_iid_data_dir(args.data_name, args.expl_iid_data_dir)
    args.data_class = get_data_class(args.data_name)
    DataClass = args.data_class
    if args.ood_factors_list is None:
        args.ood_factors_list = DataClass.DEFAULT_OOD_FACTORS_LIST
    if args.uf_groups is None:
        args.uf_groups = DataClass.DEFAULT_UF_FACTORS_LIST
    if args.loss_factors is None:
        args.loss_factors = DataClass.DEFAULT_LOSS_FACTORS
    if args.ns_shots is None:
        args.ns_shots = DataClass.DEFAULT_NS_SHOTS
    if args.mini:
        args = minify_args(args)
    if args.wandb_tags is None:
        args.wandb_tags = []
    args.wandb_tags.append(args.data_name)
    args.wandb_tags.append("DA")
    if args.coral:
        args.wandb_tags.append('coral')
    return args


def order_pivot_columns_and_rows(df,args):
    # df = df.rename(mapper=lambda col: col if not col.endswith('_mean') else (col[:-5]),axis=1)
    unordered_cols = df.columns.tolist()
    full_lf_ordering = ['c','c_mean'] + [f'c_{f}' for f in args.data_class.SHORT_FACTORS] + ['c_mx'] + [f'c_mx_{i}' for i, _ in enumerate(args.data_class.SHORT_FACTORS)] + \
                       ['x_mean', 'y_sum', 'z_sum', 'zpsi_mean'] + [f'zpsi_{cf}' for cf in args.data_class.SHORT_INTERVENED_FACTORS_AND_TRASH] + \
        ['d_sum', 'e_sum'] + [f'e_{i}' for i in range(100)] # A big enough range to cover possible dimension sizes of e
    full_uf_ordering = ['uf_all'] + [f'uf_{f}' for f in args.data_class.SHORT_INTERVENED_FACTORS_AND_TRASH] + [f'uf_{f}' for f in args.data_class.SHORT_FACTORS] + \
                       [f'uf_{i}' for i, _ in enumerate(args.data_class.SHORT_FACTORS)]
    # unique but keep order
    full_uf_ordering = list(dict.fromkeys(full_uf_ordering))
    if type(df.columns[0]) == tuple:
        ordered_cols = [ucol for lf in full_lf_ordering for uf
                        in full_uf_ordering for ucol in
                        unordered_cols if ucol == (lf, uf)] # TODO this is probably not the most efficient way to do this
    else:
        if df.columns[0] in full_lf_ordering:
            ordered_cols = [ucol for ucol in full_lf_ordering if ucol in unordered_cols]
        elif df.columns[0] in full_uf_ordering:
            ordered_cols = [ucol for ucol in full_uf_ordering if ucol in unordered_cols]
        else:
            raise ValueError(f"Unknown column {df.columns[0]}")
    unordered_rows = df.index.tolist()
    ns_shots = list(get_level_by_name(df, 'n_shots'))
    row_prefix = 'OOD_'
    ordered_rows = [urow for dis, n_shots in
                    [('ID', 0)] + [(row_prefix + f, n_shot) for f in
                                   # args.data_class.SHORT_FACTORS
                                   ['none'] + args.data_class.SHORT_OOD_VARIANTS
                                   for n_shot in ns_shots] for urow in
                    unordered_rows if (urow[0], urow[-1]) == (dis, n_shots)]

    assert len(ordered_rows) == len(unordered_rows), f"{len(ordered_rows)} != {len(unordered_rows)}"
    assert len(ordered_cols) == len(unordered_cols), f"{len(ordered_cols)} != {len(unordered_cols)}, {ordered_cols} vs {unordered_cols}"
    df = df[ordered_cols]
    df = df.reindex(ordered_rows)
    return df


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]  # https://stackoverflow.com/a/480227


def log_pivot(df, title, args):
    df = df.copy()
    df = df.astype(float)
    df = order_pivot_columns_and_rows(df,args)
    #  Log to CSV
    base_path = (wandb.run.dir.split('wandb/')[0] if not args.nowandb else wandb.run.dir)
    df.to_csv(jn(base_path, f'{title}.csv'))
    if not args.skip_plt:
        # region Log to wandb
        if 'normed_dist' in title:
            adf = df.copy()
            adf.reset_index(inplace=True)
            # if columns are multiindex, keep only the first level
            if type(adf.columns[0]) == tuple:
                adf.columns = adf.columns.get_level_values(0)
            def get_matching_c(row):
                distr = row['distribution']
                if isinstance(distr, pd.Series):
                    distr = distr.iloc[0]
                if (distr == 'ID') or (distr == 'OOD_none'):
                    # return row['c_mean']
                    # return nan
                    return float('nan')
                elif distr.startswith('OOD'):
                    shift_factor = distr.split('_')[-1]
                    return row[f'c_{shift_factor}']
                else:
                    raise ValueError

            def get_avg_nonmatching_c(row):
                distr = row['distribution']
                if distr == 'ID':
                    # return row['c_mean']
                    # return nan
                    return float('nan')
                elif distr.startswith('OOD'):
                    shift_factor = distr.split('_')[-1]
                    l0 = row.index.get_level_values(0)
                    avg_nonmatching = row[l0.str.startswith('c_') & ~(l0 == 'c_mean') & ~(l0 == 'c_matching') & ~(
                                l0 == f'c_{shift_factor}')].mean()
                    return avg_nonmatching
                else:
                    raise ValueError
            adf[('c_matching')] = adf.apply(lambda row: get_matching_c(row),axis=1)
            adf[('c_nonmatching')] = adf.apply(lambda row: get_avg_nonmatching_c(row),axis=1)
            log_per_distr_shift_cnd(adf, title)
            # reapply index
            adf.set_index(['distribution', 'n_shots'], inplace=True)
            # drop columns that are not c_mean, c_matching, c_nonmatching
            adf = adf[[('c_mean'), ('c_matching'), ('c_nonmatching')]]
            # Add row whose values are averages over all OODs
            avg_over_OODs = adf.loc[[row for row in adf.index if row[0] != 'ID']].groupby(level=1).mean()
            # the above changes the 'distribution','n_shots' index to 'n_shots' only. We add 'OOD_avg' as index
            avg_over_OODs.index = pd.MultiIndex.from_tuples([(f'OOD_avg', n_shots) for n_shots in avg_over_OODs.index],
                                                            names=['distribution', 'n_shots'])
            avg_adf = adf.query('distribution == "ID"').append(avg_over_OODs)

            avg_title = title + '_avg'
            if args.log_heatmaps:
                log_as_heatmap(avg_adf, avg_title)
                log_as_heatmap(df, title)
                uf_lf = False
                if type(df.columns[0]) == tuple:
                    uf_lf = True
                    df.columns = [l + '-' + 'u' + u.split('uf_')[1] for l, u in df.columns]
                if 'uf_' in title:
                    # As many adjacent plots as there are loss factors
                    if uf_lf:
                        loss_factors = f7([col.split("-u")[0] for col in df.columns])
                    else:
                        loss_factors = None
                    get_per_col_heatmap(df, group_by=loss_factors)
                    pc_title = title + '_per_col'
                    # plt.tight_layout()
                    wandb.log({pc_title: NonVersioningWandbImage(plt)})
                    plt.clf()

            # Log avg_adf as lineplot, . We will use a log scale for the x-axis. Because ID/0 and OOD_avg/0 would not show on a log scale, we will place ID/0 at x_axis=0.25 and OOD_avg/0 at x_axis=0.5
            avg_adf.reset_index(inplace=True)
            avg_adf['x_axis'] = avg_adf.apply(lambda row:
                                              .25 if row['distribution'] == 'ID' else
                                              .5 if ((row['distribution'] == 'OOD_avg') and (row['n_shots'] == 0))
                                              else row['n_shots'], axis=1)
            for _, row in avg_adf.iterrows():
                wandb.log({avg_title + '_c_mean': row['c_mean'], avg_title + '_c_matching': row['c_matching'], avg_title + '_c_nonmatching': row['c_nonmatching'], 'n_shots': row['x_axis']})

        # endregion
    # # Log as latex table
    # latex_path = jn(base_path, f'{title}.tex')
    # df.to_latex(latex_path, float_format='%.3f')
    # upload_to_gdrive(latex_path)

def log_as_heatmap(df, title):
    # Log avg_adf as heatmap
    sns.heatmap(df, annot=True, fmt='.3f', mask=df.isnull(),
                cmap='Reds', annot_kws={'size': 8})
    plt.yticks(rotation=0)
    fig = plt.gcf()
    n_cols = len(df.columns)
    n_rows = len(df.index)
    fig.set_size_inches((n_cols * .8 + 1.5, n_rows * .3 + 1))
    plt.tight_layout()
    wandb.log({title: NonVersioningWandbImage(plt)})
    plt.clf()


def log_per_distr_shift_cnd(adf, title):
    distr_shifts = list(dict.fromkeys(list(d for d in adf['distribution'] if d != 'ID')))
    for distr_shift in distr_shifts:
        alt_title = "per_distr_shift/" + title + '_' + distr_shift.split("_")[1]
        distr_shift_subdf = adf.query('distribution == "ID"').append(
            adf.query(f'distribution == "{distr_shift}"'))
        # Log distr_shift_subdf as lineplot, . We will use a log scale for the x-axis. Because ID/0 and OOD_avg/0 would not show on a log scale, we will place ID/0 at x_axis=0.25 and OOD_avg/0 at x_axis=0.5
        distr_shift_subdf['x_axis'] = distr_shift_subdf.apply(lambda row:
                                                              .25 if row['distribution'] == 'ID' else
                                                              .5 if ((row['distribution'] == distr_shift) and (
                                                                      row['n_shots'] == 0))
                                                              else row['n_shots'], axis=1)
        for _, row in distr_shift_subdf.iterrows():
            wandb.log({alt_title + '_c_mean': row['c_mean'], alt_title + '_c_matching': row['c_matching'],
                       alt_title + '_c_nonmatching': row['c_nonmatching'], 'n_shots': row['x_axis']})


def get_per_col_heatmap(df, group_by=None,fmt='.3f'):
    n_rows = len(df.index)
    n_cols = len(df.columns)
    if group_by is not None:
        n_color_groups = len(group_by)
    else:
        n_color_groups = n_cols

    width_ratios = get_width_ratios(df, group_by, n_color_groups)
    fig, axs = plt.subplots(1, n_color_groups, figsize=(n_cols * .8 + 1.5, n_rows * .3 + 1), gridspec_kw={'width_ratios': width_ratios})  # Squeeze the rows
    fig.subplots_adjust(wspace=0)
    cmaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys', 'YlOrBr', 'YlGnBu', 'PuRd', 'PuBuGn', 'BuPu',
             'BuGn', 'GnBu', 'OrRd', 'PuBu', 'YlGn']
    if n_color_groups == 1:
        axs = [axs]
    for i in range(n_color_groups):
        if group_by is not None:
            subdf = filter_subdf(df, group_by, i)
            cmap_idx = int((i % (n_color_groups / 2)) % len(cmaps))
        else:
            subdf = df[[df.columns[i]]]
            cmap_idx = int(i % len(cmaps))
        ax = axs[i]
        try:
            sns.heatmap(subdf, annot=True, fmt=fmt, mask=subdf.isnull(),
                        annot_kws={'size': 8}, ax=ax, cbar_kws={'location': 'top', 'format': '%.1f'},
                        yticklabels=i == 0,
                        cmap=cmaps[cmap_idx])
        except:
            print(6)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(labelsize=8)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)


def get_width_ratios(df, group_by, n_color_groups):
    width_ratios = []
    for i in range(n_color_groups):
        if group_by is not None:
            subdf = filter_subdf(df, group_by, i)
            n_sub_cols = len(subdf.columns)
            width_ratios.append(n_sub_cols)
        else:
            width_ratios.append(1)
    return width_ratios


def filter_subdf(df, group_by, i):
    if 'regex' in group_by[i]:
        subdf = df.filter(regex=group_by[i]['regex'])
    else:
        subdf = df.filter(like=group_by[i])
    return subdf


def get_loss_df_csv_dir(run_name):
    base_path = jn(CITRIS_ROOT,
                   'DA_logs',
                   run_name)
    loss_df_csv_dir = get_versioned_path(base_path)
    print("Saving loss df to", loss_df_csv_dir)
    os.makedirs(loss_df_csv_dir, exist_ok=True)
    return loss_df_csv_dir


def set_up_dfs(args):
    row_idx_dict = {}
    maybe_ID = [] if args.skip_ID else ['ID']
    row_idx_dict['distribution'] = maybe_ID + ['OOD_' + '_'.join(el) for el in args.ood_factors_list]
    maybe_0shot = [] if args.skip_0shot else [0]
    row_idx_dict['n_shots'] = (maybe_0shot + args.ns_shots) if not (0 in args.ns_shots) else args.ns_shots
    model_idx = []
    model_id2ckpt_path = {}
    model_id2model_class = {}
    model_id2loss_factors = {}
    model_id2param_groups = {}
    model_id2is_nan = {}
    for wid, ckpt_path, model_class in zip(args.wandb_ids, args.ckpt_paths, args.model_classes):
        # Need to load the ckpt to get some info about its hyperparams
        ckpt_args = {
            'checkpoint_path': ckpt_path,
            'DataClass': args.data_class,
            'beta_coral': args.beta_coral,
            'strict': False
        }
        add_maybe_override_args(args, ckpt_args)
        model = model_class.load_from_checkpoint(**ckpt_args)
        col_name_prefix = get_col_name_prefix(model)
        col_name = col_name_prefix + wid
        model_idx.append(col_name)
        model_id2ckpt_path[col_name] = ckpt_path
        model_id2model_class[col_name] = model_class
        model_id2loss_factors[col_name] = model.DA_loss_factor_keys
        model_id2param_groups[col_name] = model.get_grouped_flp_params(args.use_maxcorr_psi).keys() if not args.coral else None
        isnan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any().item()
        if isnan:
            args.wandb_tags.append('NaN')
        model_id2is_nan[col_name] = isnan
    # assert len(model_id2model_class.values()) == len(
    #     set(model_id2model_class.values())), "Multiple models of the same class are not supported, for easier pretty wandb logging"
    # row_idxs.append(model_idx)
    row_idx_dict['model'] = model_idx
    # if args.piecewise_unfreeze:
    uf_groups = args.uf_groups
    if (not args.skip_uf_all) and ['all'] not in uf_groups:
        uf_groups = [['all']] + uf_groups
    row_idx_dict['unfreeze_factor'] = list(dict.fromkeys(
        ['uf_' + '_'.join([str(e) for e in el]) for el in uf_groups]))
    # args.coarse_loss_factors = list(dict.fromkeys([SHORT_FINE2COARSE_old[f] for f in args.loss_factors]))
    # row_idx_dict['loss_factor'] = (['c_mean'] + ['c_' + el for el in (args.loss_factors)] + # TLP, MTLP
    #                               ['c_mx_mean'] + [f'c_mx_{SHORT_FACTORS_old.index(f)}' for f in args.loss_factors] + # MTLP
    #                               ['zpsi_mean'] + [f'zpsi_t'] + [f'zpsi_{f}' for f in args.coarse_loss_factors] + # *TRIS
    #                               ['x_mean'] + ['y_sum'] + # *TRIS
    #                               ['z_sum'] + # *TRIS, MFP
    #                               ['d_sum'] ) # MFP
    travel_idx_dict = row_idx_dict.copy()
    row_idx_dict['loss_factor'] = list(dict.fromkeys([lf for lfks in model_id2loss_factors.values() for lf in lfks]))
    row_idx_dict['metric'] = args.metrics
    row_idx = pd.MultiIndex.from_product(row_idx_dict.values(), names=row_idx_dict.keys())
    loss_df = pd.DataFrame(columns=['value'], index=row_idx)
    loss_df = trim_lossdf_rows(loss_df, row_idx_dict, model_id2loss_factors, args)

    if not args.coral:
        travel_idx_dict['param_group'] = list(dict.fromkeys([pg for pgs in model_id2param_groups.values() for pg in pgs]))
        travel_df = pd.DataFrame(columns=['value'], index=pd.MultiIndex.from_product(travel_idx_dict.values(), names=travel_idx_dict.keys()))
        travel_df = trim_traveldf_rows(travel_df, travel_idx_dict, model_id2param_groups, args)
    else:
        travel_df = None

    return loss_df, travel_df, model_id2ckpt_path, model_id2model_class, model_id2is_nan


def trim_lossdf_rows(loss_df, row_idx_dict, model_id2loss_factors, args):
    loss_df = loss_df.reset_index()
    # Remove more than 0 shots for ID
    loss_df = loss_df[(loss_df.distribution != 'ID') | (loss_df.n_shots == 0)]
    # Remove nll, logstd, minlogstd and maxlogstd for x_mean
    loss_df = loss_df[~((loss_df.metric.isin(['nll', 'logstd', 'minlogstd', 'maxlogstd'])) & (loss_df.loss_factor.isin(['x_mean'])))]
    # Remove mse, mae, logstd, minlogstd and maxlogst for y_sum, d_sum
    loss_df = loss_df[~((loss_df.metric.isin(['mse', 'mae', 'logstd', 'minlogstd', 'maxlogstd'])) & (loss_df.loss_factor.isin(['y_sum','d_sum'])))]\
    # Keep only dist for c_*
    loss_df = loss_df[~loss_df.loss_factor.str.startswith('c_') | (loss_df.metric == 'normed_dist')]
    # Remove dist for non c_*
    loss_df = loss_df[~(loss_df.loss_factor.str.startswith('c_') == False) | (loss_df.metric != 'normed_dist')]

    # region trimming based on model and loss factor
    # Keep only loss factors that go with model according to model_id2loss_factors
    loss_df = loss_df[loss_df.apply(lambda row: row.loss_factor in model_id2loss_factors[row.model], axis=1)]
    # endregion

    loss_df = trim_model_vs_uf(args, loss_df)

    loss_df.set_index(list(row_idx_dict.keys()), inplace=True)
    return loss_df


def trim_model_vs_uf(args, df):
    df = df[(df.unfreeze_factor == 'uf_all') |
            (df.unfreeze_factor == 'uf_all_enc') |
            (df.unfreeze_factor == 'uf_cheat_pnu') |
            (df.unfreeze_factor == 'uf_pred_pnu') |
            # Check if part from uf_factor after '_' is a digit and model starts with MTLP or contains MFP
            (df.unfreeze_factor.str.split('_').str[-1].str.isdigit() & (df.model.str.startswith(
                          'MTLP') | df.model.str.contains('MFP'))) |
            # Check if uf_factor ends with element of SHORT_FACTORS and model starts with TLP
            (df.unfreeze_factor.str.split('_').str[-1].isin(
                          args.data_class.SHORT_FACTORS) & df.model.str.startswith('TLP')) |
            # Check if uf_factor ends with element of SHORT_INTERVENED_FACTORS_AND_TRASH and model contains TRIS
            (df.unfreeze_factor.str.split('_').str[-1].isin(
                          args.data_class.SHORT_INTERVENED_FACTORS_AND_TRASH) & df.model.str.contains('TRIS'))]
    return df


def trim_traveldf_rows(travel_df, travel_idx_dict, model_id2param_groups, args):
    travel_df = travel_df.reset_index()
    # Remove ID # TODO maybe DON'T trim this for coral?
    travel_df = travel_df[travel_df.distribution != 'ID']
    # Remove 0 shots
    travel_df = travel_df[travel_df.n_shots != 0]
    # Keep only param_groups that go with model according to model_id2param_groups
    travel_df = travel_df[travel_df.apply(lambda row: row.param_group in model_id2param_groups[row.model], axis=1)]

    travel_df = trim_model_vs_uf(args, travel_df)
    travel_df.set_index(list(travel_idx_dict.keys()), inplace=True)
    return travel_df


def get_col_name_prefix(model):
    if type(model) == CITRISNF:
        col_name_prefix = 'TRISNF_'
        if model.hparams.fullview_baseline:
            if model.hparams.beta_classifier == 0:
                col_name_prefix = 'St' + col_name_prefix
            else:
                col_name_prefix = 'FV' + col_name_prefix
        else:
            col_name_prefix = 'CI' + col_name_prefix
        if model.hparams.skip_flow:
            col_name_prefix = 'SF' + col_name_prefix
        elif model.hparams.random_flow:
            col_name_prefix = 'RF' + col_name_prefix
        if (model.hparams.lambda_reg == 0) and (model.hparams.beta_classifier == 0):
            col_name_prefix = col_name_prefix.replace('TRIS','_RVP')
        if model.hparams.flow_wid is not None:
            col_name_prefix = "CC" + col_name_prefix
    elif type(model) == CITRISVAE:
        col_name_prefix = 'TRISVAE_'
        if model.hparams.fullview_baseline:
            if model.hparams.beta_classifier == 0:
                col_name_prefix = 'St' + col_name_prefix
            else:
                col_name_prefix = 'FV' + col_name_prefix
        else:
            col_name_prefix = 'CI' + col_name_prefix
    elif type(model) == OldMixedTrueLatentPrior:
        if model.hparams.mix:
            col_name_prefix = 'MTLP_'
        else:
            col_name_prefix = 'TLP_'
    elif type(model) == MixedFactorPredictor:
        if model.fixed_encoder == 'perfect_ux':
            col_name_prefix = 'PMFP_'
        elif model.fixed_encoder == 'identity':
            col_name_prefix = 'IMFP_'
        elif ('anti_align' in model.hparams) and model.hparams.anti_align:
            col_name_prefix = 'AAMFP_'
        else:
            col_name_prefix = 'MFP_'
    else:
        raise ValueError(f"Unknown model class {type(model)}")
    return col_name_prefix


def get_versioned_path(base_path):
    existing_versions = glob(base_path + '_v*')
    if len(existing_versions) > 0:
        version = max([int(re.search(r'_v(\d+)', v).group(1)) for v in existing_versions]) + 1
        versioned_path = base_path + f'_v{version}'
    else:
        versioned_path = base_path + '_v0'
    return versioned_path


def set_pandas_full_print():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


def ttest_da(args, model, test_loader,distribution=None):
    model.eval()
    results_dict = {}
    for plot_order, batch in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            # batch = [el.to(CUDA_DEVICE) if el != [] else [] for el in batch] # can be empty list if require_imgs is False
            batch = CUDAfy(batch)
            loss_dict = model.get_losses(batch, args=args)
            if plot_order == 0:
                for k in loss_dict.keys():
                    results_dict[k] = []
            for k in loss_dict.keys():
                results_dict[k].append(loss_dict[k])
            if args.mini:
                break
    mean_result_dict = {k:
                            (tn(torch.stack(v).amin(0)) if ('minlogstd' in k) else
                             tn(torch.stack(v).amax(0)) if ('maxlogstd' in k)
                             else tn(torch.cat(v, dim=0).flatten(0,1)) if ('_per_hyp' in k) # B(L-1)_S0_Z_(C+1) -> B(L-1)S0_Z_(C+1)
                             else tn(torch.stack(v).mean(0)))
                        for k, v in results_dict.items()}

    return mean_result_dict


def train_da(args, model, n_shots, optimizer, scheduler_dict, train_loader, val_loader=None, unfreeze_setting='uf_all', distribution=None, pretrain_phase=False):
    print(f"{'Pret' if pretrain_phase else 'T'}raining model {model._get_name()} with {n_shots} shots and {unfreeze_setting} unfreeze_subset for {distribution} distribution")
    if not pretrain_phase:
        model.prep_for_fsl_train()
        mask_dict = model.get_masks_for_unfreeze(unfreeze_setting, train_loader, distribution, args)
    if scheduler_dict is not None:
        scheduler = scheduler_dict['scheduler']
        assert scheduler_dict['interval'] == 'step'
    log_name = get_log_name(distribution, model._get_name(), n_shots, unfreeze_setting, args.wandb_ids)
    if not (args.coral or pretrain_phase):
        pre_adapt_params = model.get_grouped_flp_params() #TODO if psi changes enough so that its samples differ, the groups in this will differ from the post_adapt_params
        # pre_adapt_psi = model.psi.clone() # TODO I probably want to use a cheating psi instead of learned psi for models with bad psi
        pre_adapt_psi = model.get_learnt_or_maxcorr_psi(use_maxcorr=args.use_maxcorr_psi)
    old_stuff(args, model)

    do_validation = (val_loader is not None) and (len(val_loader) > 0)
    if do_validation:
        best_val_loss = validate_da(args, batch_idx=0, epoch=0, log_name=log_name, model=model, val_loader=val_loader, pretrain_phase=pretrain_phase)
        best_val_model = copy.deepcopy(model)
        best_val_epoch = 0
    # train set performance before training
    # if not args.skip_0ep_train_eval:
    if 'skip_0ep_train_eval' not in args.__dict__ or not args.skip_0ep_train_eval:
        with torch.no_grad():
            mean_loss_dict = {}
            for batch_idx, batch in tqdm(enumerate(train_loader)):
                insert_or_add(get_train_loss(batch, model, args, pretrain_phase=pretrain_phase), mean_loss_dict)
        divide_dict_by_count(len(train_loader), mean_loss_dict)
        # # clear memory
        # torch.cuda.empty_cache()
        log_fsl_results(args, epoch=0, batch_idx=0, log_name=log_name, loss_dict=mean_loss_dict, data_loader=train_loader, split='train' if not pretrain_phase else 'pretrain')
    for epoch in (pbar := tqdm(range(args.num_epochs))):

        mean_loss_dict = {}
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            loss_dict = get_train_loss(batch, model, args, pretrain_phase=pretrain_phase)
            optimizer.zero_grad()
            backprop_key = 'fsl_train_loss' if not pretrain_phase else 'ID_loss'
            loss_dict[backprop_key].backward()
            if not pretrain_phase:
                old_param_dict = {n: p.clone() for n, p in model.get_sparse_update_params()}
            optimizer.step()
            if not pretrain_phase:
                for n, new_param in model.get_sparse_update_params():
                    new_param.data = torch.where(mask_dict[n].to(torch.bool), new_param.data, old_param_dict[n].data)
            if scheduler_dict is not None:
                scheduler.step()
            with torch.no_grad():
                insert_or_add(loss_dict, mean_loss_dict)
            # if args.per_batch_logging:
            if hasattr(args, 'per_batch_logging') and args.per_batch_logging:
                log_fsl_results(args, epoch+1, batch_idx, log_name, loss_dict, train_loader, 'train' if not pretrain_phase else 'pretrain')
        divide_dict_by_count(len(train_loader), mean_loss_dict)
        log_fsl_results(args, epoch+1, batch_idx, log_name, mean_loss_dict, train_loader, 'train' if not pretrain_phase else 'pretrain')
        pbar.set_description(f"loss {mean_loss_dict['fsl_train_loss'].item():.3f}", refresh=True)
        if do_validation and (args.check_val_every_n_epoch > 0) and (epoch % args.check_val_every_n_epoch == 0) and (epoch < (args.num_epochs-1)):
            best_val_loss, best_val_model, updated = maybe_update_best_val(args, batch_idx, best_val_loss, best_val_model,  epoch, log_name, model, val_loader, pretrain_phase=pretrain_phase)
            if updated:
                best_val_epoch = epoch
    if do_validation:
        best_val_loss, best_val_model, updated = maybe_update_best_val(args, batch_idx, best_val_loss, best_val_model, epoch, log_name, model, val_loader, pretrain_phase=pretrain_phase)
        if updated:
            best_val_epoch = epoch
    else:
        best_val_model = model
        best_val_epoch = epoch

    #
    # avg_param_travel = get_avg_param_travel(model, pre_adapt_params, pre_adapt_psi, args)
    # best_val_avg_param_travel = get_avg_param_travel(best_val_model, pre_adapt_params, pre_adapt_psi, args)
    # return {
    #     'last': {'model': model, 'avg_param_travel': avg_param_travel},
    #     'best_val': {'model': best_val_model, 'avg_param_travel': best_val_avg_param_travel, 'epoch': best_val_epoch}
    # }
    result = {
        'last': {'model': model},
        'best_val': {'model': best_val_model, 'epoch': best_val_epoch}
    }
    if not args.coral:
        result['last']['avg_param_travel'] = get_avg_param_travel(model, pre_adapt_params, pre_adapt_psi, args)
        result['best_val']['avg_param_travel'] = get_avg_param_travel(best_val_model, pre_adapt_params, pre_adapt_psi, args)
    return result
    # max_val = max([v.max() for k, v in param_travel.items()])
    # min_val = min([v.min() for k, v in param_travel.items()])
    # normed_param_travel = {k: (v - min_val) / (max_val - min_val) for k, v in param_travel.items()}
    # per_factor_per_layer_travel = {i : {k: v[i].mean() for k, v in normed_param_travel.items()} for i in range(model.num_fine_vars)}
    # per_factor_travel = {i: sum([travel for layer, travel in layer2travel.items()])/len(layer2travel) for i, layer2travel in per_factor_per_layer_travel.items()}
    # df = pd.DataFrame(per_factor_per_layer_travel).transpose()
    # arr = df.to_numpy()

    # return model, avg_param_travel


def old_stuff(args, model):
    if args.reset_matching:
        raise NotImplementedError("reset_matching not implemented")  # below doesn't change the params
        # if unfreeze_subset == 'uf_all':
        #     # Detect which factor changed based on for which dimension the NLL of the pretrained model is highest
        #     # Then reset the group of flp parameters corresponding to that factor
        #     with torch.no_grad():
        #         loss_dict = get_train_loss(next(iter(train_loader)), model, args)
        #     max_idx = int(loss_dict['nll_per_e'].argmax())
        # else:
        #     max_idx = model.get_unfreeze_idx(unfreeze_subset)
        # params_for_max_loss = model.get_grouped_flp_params(clone=False)[max_idx] # list of tensors
        # for param in params_for_max_loss:
        #     param.data = torch.randn_like(param)
    if (hasattr(model, 'cma_true_parentage') and model.cma_true_parentage):  # TODO also for predicted parentage
        raise NotImplementedError("Resetting nonparent-mask for cma_true_parentage not implemented")
        # if unfreeze_subset == 'uf_all':
        #     # Detect which factor changed based on for which dimension the NLL of the pretrained model is highest
        #     # Then reset the group of flp parameters corresponding to that factor
        #     with torch.no_grad():
        #         loss_dict = get_train_loss(next(iter(train_loader)), model, args)
        #     max_idx = int(loss_dict['nll_per_e'].argmax())
        # else:
        #     max_idx = model.get_unfreeze_idx(unfreeze_subset)
        model.reset_matching_parent_mask(max_idx)


def get_avg_param_travel(model, pre_adapt_params, pre_adapt_psi, args):
    # post_adapt_psi = model.prior_t1.get_target_assignment(hard=True) if hasattr(model, 'prior_t1') else None
    # post_adapt_psi = model.psi
    post_adapt_psi = model.get_learnt_or_maxcorr_psi(use_maxcorr=args.use_maxcorr_psi)
    # Check if int tensors pre_adapt_psi and post_adapt_psi are equal
    if (pre_adapt_psi is None) or torch.equal(pre_adapt_psi, post_adapt_psi):
        param_travel = {k: [tn(abs(v - pre_adapt_params[k][i])) for i, v in enumerate(lst)] for k, lst in
                        model.get_grouped_flp_params().items()}
        avg_param_travel = {k: sum([v.mean() for i, v in enumerate(lst)]) / len(lst) if (len(lst) != 0) else [] for
                            k, lst in param_travel.items()}
    else:
        print(f"pre_adapt_psi != post_adapt_psi, so not logging param_travel")
        avg_param_travel = {}
    return avg_param_travel


def maybe_update_best_val(args, batch_idx, best_val_loss, best_val_model, epoch, log_name, model, val_loader, pretrain_phase=False):
    new_val_loss = validate_da(args, batch_idx, epoch + 1, log_name, model, val_loader, pretrain_phase=pretrain_phase)
    updated = False
    if new_val_loss < best_val_loss:
        best_val_loss, best_val_model = new_val_loss, copy.deepcopy(model)
        updated = True
    return best_val_loss, best_val_model, updated


def divide_dict_by_count(count, mean_loss_dict):
    for k, v in mean_loss_dict.items():
        mean_loss_dict[k] = v / count


def insert_or_add(loss_dict, mean_loss_dict):
    for k, v in loss_dict.items():
        if k not in mean_loss_dict:
            mean_loss_dict[k] = v.detach()#.cpu().detach().numpy()
        else:
            mean_loss_dict[k] += v.detach()#.cpu().detach().numpy()


def validate_da(args, batch_idx, epoch, log_name, model, val_loader, pretrain_phase=False):
    mean_loss_dict = {}
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            loss_dict = get_train_loss(batch, model, args, pretrain_phase=pretrain_phase)
            insert_or_add(loss_dict, mean_loss_dict)            # Check if model class is (subclass of) CITRISVAE
            if isinstance(model, CITRISVAE) and socket.gethostname() == "theoden":
                torch.cuda.empty_cache() # A very hacky way to avoid CUDA out of memory error on theoden
        divide_dict_by_count(len(val_loader), mean_loss_dict)
        log_fsl_results(args, epoch, batch_idx, log_name, mean_loss_dict, val_loader, 'val' if not pretrain_phase else 'preval')
        model.prep_for_fsl_train()
    return loss_dict['fsl_train_loss']


def log_fsl_results(args, epoch, batch_idx, log_name, loss_dict, data_loader, split='train'):
    # log the train loss to wandb, identify chart by model name, n_shots, unfreeze_factor, distribution
    log_name_and_split = f'{log_name}_{split}'
    dct_to_log = {f'{log_name_and_split}_loss': loss_dict['fsl_train_loss'],
               'epoch': epoch,
               'batch_per_epoch': batch_idx,
               'total_batch': epoch * len(data_loader) + batch_idx}
    if 'l_coral' in loss_dict and split in ['pretrain','preval']:
        dct_to_log[f'{log_name_and_split}_l_coral'] = loss_dict['l_coral']
        dct_to_log[f'{log_name_and_split}_ID_loss'] = loss_dict['ID_loss']
    wandb.log(dct_to_log)
    if args.log_all_train_values:
        log_da_to_wandb(epoch, batch_idx, log_name_and_split, loss_dict, data_loader)
    else:
        for key in ['minlogstd_et1', 'mse_dt1_mean', 'normed_dist_c_mean', 'mse_et1_mean']:
            if key in loss_dict:
                wandb.log({f'{log_name_and_split}{key}': loss_dict[key],
                           'epoch': epoch,
                            'batch_per_epoch': batch_idx,
                            'total_batch': epoch * len(data_loader) + batch_idx})


def get_log_name(distribution, model_name, n_shots, unfreeze_factor, w_ids):
    log_name = 'per_distr_shift/'
    if len(w_ids) > 1:
        log_name += model_name + '-'
    # if len(args.uf_groups) > 1:
    #     log_name += unfreeze_factor + '-'
    log_name += unfreeze_factor + '-'
    log_name += f'{distribution}-{n_shots}sh-'
    return log_name


def log_da_to_wandb(epoch, batch_idx, log_name, loss_dict, loader):
    for k, v in loss_dict.items():
        if '_per_' in k:
            groups = re.search(r'(.*)_per_(.*)', k)
            metric = groups.group(1)
            dim = groups.group(2)
            # as dict comprehension
            wandb.log({f'{log_name}{metric}_{dim}_{idx}': val for idx, val in enumerate(v)} | {'epoch': epoch, 'batch': batch_idx, 'total_batch': epoch * len(loader) + batch_idx})
            # for idx, val in enumerate(v):
            #     wandb.log({log_name + f'{metric}_{dim}_{idx}': val, 'epoch': epoch, 'batch_per_epoch': i,
            #                'total_batch': epoch * len(train_loader) + i})
        else:
            wandb.log(
                {log_name + k: v, 'epoch': epoch, 'batch_per_epoch': batch_idx, 'total_batch': epoch * len(loader) + batch_idx})


def get_train_loss(batch, model, args=None, pretrain_phase=False):
    batch = CUDAfy(batch)
    loss_dict = model.get_losses(batch, args=args, pretrain_phase=pretrain_phase)
    # train_loss = model.get_FSL_train_loss(batch)
    return loss_dict


def get_iid_data_dir(data_name, explicit_iid_data_dir=None):
    if explicit_iid_data_dir is None:
        if data_name == 'pong':
            return PONG_ROOT
        elif (data_name == 'causal3d') or (data_name == 'shapes'):
            return TC3DI_ROOT
        else:
            raise ValueError(f'Unknown data name {data_name}')
    else:
        return explicit_iid_data_dir


def args_to_ckpt_paths_and_model_classes_and_data_name(args):
    '''
    Deduces the ckpt paths, model classes and data name from the args.wandb_ids, unless explicitly specified by expl_ckpt_paths, expl_model_classes, and expl_data_name
    '''
    if not args.ckpt_paths:
        if not args.wandb_ids:
            raise ValueError("Must provide either ckpt_paths or wandb_ids")
        else:
            # Use best path instead of last. Best is stored with epoch=*-step=*.ckpt
            if args.expl_ckpt_paths is None:
                if args.pt_epoch is not None:
                    args.ckpt_paths = [glob(jn(CITRIS_CKPT_ROOT, f"{wandb_id}_*/manual_epoch_{args.pt_epoch}.ckpt"))[0] for wandb_id in
                                   args.wandb_ids]
                else:
                    args.ckpt_paths = [glob(jn(CITRIS_CKPT_ROOT, f"{wandb_id}_*/epoch=*-step=*.ckpt"))[0] for wandb_id in
                                   args.wandb_ids]
            else:
                args.ckpt_paths = args.expl_ckpt_paths
            # args.model_classes = [eval(model_class_name) for model_class_name in
            #                       [re.search(f'{wid}_(.+)/epoch=.*-step=.*.ckpt', text).group(1) for wid, text in
            #                        zip(args.wandb_ids, args.ckpt_paths)]]
            if args.expl_model_classes is None:
                args.model_classes = [eval(model_class_name) for model_class_name in
                                      [re.search(f'{wid}_(.+)_.*/.*\\.ckpt', text).group(1) for wid, text in
                                       zip(args.wandb_ids, args.ckpt_paths)]]
            else:
                args.model_classes = [eval(mcn) for mcn in args.expl_model_classes]
            if args.expl_data_name is None:
                data_names = [re.search(f'{wid}_.+_(.+)/.*\\.ckpt', text).group(1) for wid, text in
                                 zip(args.wandb_ids, args.ckpt_paths)]
                # ensure that data names are all the same
                assert len(set(data_names)) == 1
                args.data_name = data_names[0]
            else:
                args.data_name = args.expl_data_name
    else:
        raise NotImplementedError(
            "providing ckpt_paths conflicts with naming of df columns, and haven't implemented non-conflicting option")
        if args.wandb_ids:
            print("Ignoring wandb_ids because ckpt_paths were provided")


def load_few_shot_datasets(args, distribution, n_shots):
    '''
    if coral:
        if distribution == 'id': 
            return { 'train': id_train_loader, 'val': id_val_loader, ' test': id_test_loader}
        else:
            if n_shots == 0:
                return { 'train': id_train_loader, 'val': id_val_loader, ' test': ood_test_loader}
            else:
                return { 'train': idOod_train_loader, 'val': idOod_val_loader, ' test': ood_test_loader}
    else:
        if distribution == 'id':
            return { 'test': id_test_loader}
        else:
            if n_shots == 0:
                return { 'test': ood_test_loader}
            else:
                return { 'train': ood_train_loader, 'val': ood_val_loader, ' test': ood_test_loader}
    '''
    print(f"Loading few-shot datasets for {distribution} distribution")
    start = time.time()
    DataClass = args.data_class

    if distribution == 'ID' or n_shots == 0:
        id_train_loader, id_val_loader, id_test_loader = [get_loader(args, DataClass, args.iid_data_dir, split) for split in ['train', 'val', 'test']]
    if distribution != 'ID':
        ood_data_dir, split2permutation = get_ood_data_dir_and_permutation(DataClass, args, distribution)
        if n_shots > 0:
            # split2idxs = {split: split2permutation[split][:n_shots] for split in split2permutation}
            split2idxs = {
                'train': split2permutation['train'][:round(n_shots * args.train_val_split)],
                'val': split2permutation['train'][round(n_shots * args.train_val_split):n_shots]
            }
            if split2idxs['val'] == []:
                print(f'Warning: n_shots={n_shots} and train_val_split={args.train_val_split}, so val set is empty')
            if args.coral:
                idOod_train_loader, idOod_val_loader = [get_loader(args, DataClass, args.iid_data_dir, split,
                                                                   second_data_folder=ood_data_dir,
                                                                   subset_idxs=split2idxs[split]) for split in ['train', 'val']]
            ood_train_loader, ood_val_loader = [get_loader(args, DataClass, ood_data_dir, split,
                                                           subset_idxs=split2idxs[split], mini=False, get_val_from_train=True) for split in ['train', 'val']]
        ood_test_loader = get_loader(args, DataClass, ood_data_dir, 'test', mini=False)
    print(f"Finished loading few-shot datasets in {time.time() - start:.1f} seconds")
    if args.coral:
        if distribution == 'ID':
            return {'pretrain': id_train_loader, 'preval': id_val_loader, 'test': id_test_loader}
        else:
            if n_shots == 0:
                return {'train': id_train_loader, 'val': id_val_loader, 'test': ood_test_loader}
            else:
                # return {'train': idOod_train_loader, 'val': idOod_val_loader, 'test': ood_test_loader}
                return {'pretrain': idOod_train_loader, 'preval': idOod_val_loader, 'train': ood_train_loader, 'val': ood_val_loader, 'test': ood_test_loader}
    else:
        if distribution == 'ID':
            return {'test': id_test_loader}
        else:
            if n_shots == 0:
                return {'test': ood_test_loader}
            else:
                return {'train': ood_train_loader, 'val': ood_val_loader, 'test': ood_test_loader}
    #      fullid_fs_ood_train_dataset = FullID_FewShotOOD_Dataset(train_dataset, full_ood_train_dataset, idxs)
    #     fullid_fs_ood_val_datset = FullID_FewShotOOD_Dataset(val_tlp_dataset, full_ood_train_dataset, idxs)
    #     fullid_fs_ood_test_datset = FullID_FewShotOOD_Dataset(test_tlp_dataset, full_ood_train_dataset, idxs)



    # return_dict = {}
    # DataClass = args.data_class
    # if distribution == 'ID':
    #     id_test_dataset = DataClass(data_folder=args.iid_data_dir, split='test', coarse_vars=True,
    #                                        return_latents=True)
    #     return_dict |= {'test': data.DataLoader(id_test_dataset, batch_size=args.test_batch_size,
    #                                     shuffle=True, pin_memory=True, num_workers=args.num_workers)}
    #     if args.coral:
    #         id_train_dataset = DataClass(data_folder=args.iid_data_dir, split='train', coarse_vars=True,
    #                                            return_latents=True)
    #         id_val_dataset = DataClass(data_folder=args.iid_data_dir, split='val', coarse_vars=True,
    #                                          return_latents=True)
    #         return_dict |= {'train': data.DataLoader(id_train_dataset, batch_size=args.batch_size,
    #                                             shuffle=True, pin_memory=True, num_workers=args.num_workers),
    #                         'val': data.DataLoader(id_val_dataset, batch_size=args.test_batch_size,
    #                                           shuffle=True, pin_memory=True, num_workers=args.num_workers)}
    # else:
    #     ood_data_dir, permutation = get_ood_data_dir_and_permutation(DataClass, args, distribution)
    #     train_dataset = DataClass(data_folder=ood_data_dir, split='train', coarse_vars=True, return_latents=True)
    #     val_dataset = DataClass(data_folder=ood_data_dir, split='val', coarse_vars=True, return_latents=True)
    #     test_dataset = DataClass(data_folder=ood_data_dir, split='test', coarse_vars=True, return_latents=True)
    #
    #     if not args.coral:
    #         return_dict |= {'train': data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                             shuffle=True, pin_memory=True, num_workers=args.num_workers),
    #                         'val': data.DataLoader(val_dataset, batch_size=args.test_batch_size,
    #                                           shuffle=True, pin_memory=True, num_workers=args.num_workers),
    #                         'test': data.DataLoader(test_dataset, batch_size=args.test_batch_size,
    #                                            shuffle=True, pin_memory=True, num_workers=args.num_workers)}
    #     else:
    #         # fullid_fs_ood_train_dataset = FullID_FewShotOOD_Dataset(train_dataset, full_ood_train_dataset, idxs)
    #         # data_loaders['train'] = data.DataLoader(fullid_fs_ood_train_dataset, sampler=BatchSampler(RandomSampler(train_dataset),batch_size=args.batch_size,drop_last=False),num_workers=args.num_workers)
    # return return_dict


def get_ood_data_dir_and_permutation(DataClass, args, distribution):
    ood_factors = distribution.split('OOD_')[1].split('__')
    ood_factors_long = [args.data_class.SHORT2F[long_factor] if long_factor != 'none' else 'none' for long_factor in ood_factors]
    ood_root_dir = get_ood_root_dir_for_data_class(DataClass)
    # with open(jn(ood_root_dir, "idx_permutations.json"), 'r')  as f: # generated with create_ood_idx_permutations_for_fsl.py
    #     idx_permutations = json.load(f)
    # permutation = idx_permutations[str(args.fsl_seed)]
    split2permutation = {}
    for split in ['train', 'val']:
        with open(jn(ood_root_dir, f"idx_permutations_{split}.json"), 'r')  as f:
            idx_permutations = json.load(f)
        split2permutation[split] = idx_permutations[str(args.fsl_seed)]
    data_dir = jn(ood_root_dir, "__".join(ood_factors_long))
    return data_dir, split2permutation


def upload_to_gdrive(tex_file_path):
    try:
        file_name = os.path.basename(tex_file_path)
        latex_folder_id = '1_lRSfbyBbVoqz-d6rjcfrMw2J6lfxNSH'
        file_list = drive.ListFile({'q': f"'{latex_folder_id}'  in parents and trashed = False"}).GetList()

        file_id = None
        for x in range(len(file_list)):
            if file_list[x]['title'] == file_name:
                file_id = file_list[x]['id']
                break

        if file_id is None:
            gfile = drive.CreateFile({'title': file_name, 'parents': [{'id': latex_folder_id}]})
        else:
            gfile = drive.CreateFile({'id': file_id})
        # Read file and set it as the content of this instance.
        gfile.title = file_name
        gfile.SetContentFile(tex_file_path)
        gfile.Upload()  # Upload the file.
    except Exception as e:
        print(e)
        print("Error uploading latex file, continuing anyway")



def get_diff_subdf(full_loss_df, absolute=True):
    full_loss_df = full_loss_df[
        (full_loss_df['n_shots'] == 0) & (full_loss_df['metric'] == 'normed_dist') & (full_loss_df['unfreeze_factor'] == 'uf_all')]
    # remove columns 'model', 'unfreeze_factor', 'metric', 'n_shots'
    full_loss_df = full_loss_df.drop(columns=['model', 'unfreeze_factor', 'metric', 'n_shots'])
    # This is a df with columns: distribution, n_shots, model, unfreeze_factor, loss_factor, metric, value
    # Make pivot df with row multi-index: distribution, n_shots, model, unfreeze_factor, metric; with column index: loss_factor; with values: value
    pivot_df = full_loss_df.pivot_table(index=['distribution'], columns='loss_factor', values='value')
    # move 'c_mean' to first column: concatenate ['c_mean'] with the rest of the columns
    pivot_df = pivot_df[['c_mean'] + [c for c in pivot_df.columns if c != 'c_mean']]
    # rows look like: ID, OOD_a, OOD_hb, ... . cols look like: c_mean, c_a, c_hb, ...
    # diff_df: difference in value between OOD_a and ID, OOD_hb and ID, ...
    diff_df = pivot_df - pivot_df.loc['ID']
    if not absolute:
        diff_df /= pivot_df.loc['ID']
    # add columns: c_match (if c_* matches OOD_*), c_nonmatch (if c_* does not match OOD_*)
    diff_df['c_match'] = diff_df.apply(get_matching_c, axis=1)
    diff_df['c_nonmatch'] = diff_df.apply(get_nonmatching_c, axis=1)
    # add row: avg of OOD_* rows
    diff_df.loc['OOD_avg'] = diff_df.loc[[d for d in diff_df.index if d.startswith('OOD')]].mean()
    # subdf: keep row OOD_avg and columns c_match, c_nonmatch, c_mean
    subdf = diff_df.loc[['OOD_avg'], ['c_match', 'c_nonmatch', 'c_mean']]
    return subdf


def get_matching_c(row):
    distr = row.name
    if distr == 'ID':
        return row['c_mean']
    elif distr.startswith('OOD'):
        shift_factor = distr.split('_')[-1]
        return row[f'c_{shift_factor}']
    else:
        raise ValueError


def get_nonmatching_c(row):
    distr = row.name
    if distr == 'ID':
        return row['c_mean']
    elif distr.startswith('OOD'):
        shift_factor = distr.split('_')[-1]
        # average of non-matching c's
        # return row[[c for c in row.index if c != f'c_{shift_factor}']].mean()
        # average of c's that are not c_mean, c_matching, or that match the shift factor
        return row[[c for c in row.index if c not in ['c_mean', f'c_{shift_factor}', 'c_match']]].mean()
    else:
        raise ValueError


if __name__ == '__main__':
    main()


def add_oodlog_callback_kwargs(args, callback_kwargs, datasets, model_class):
    if args.do_ood_logging:
        assert args.ood_log_nsshots is not None
        ood_args = Namespace(**{
            'data_class': datasets['train'].__class__,
            'iid_data_dir': args.data_dir,
            'mini': args.mini,
            'model_classes': [model_class],
            'num_workers': args.num_workers,
            'batch_size': args.batch_size,
            'test_batch_size': args.batch_size,
            'fsl_seed': args.seed % 100,
            # Kinda hacky to hardcode 100, but jn(ood_root_dir, f"idx_permutations_{split}.json") has 100 options
            'train_val_split': args.ood_train_val_split,
            'coral': False,
        })
        shifts = datasets['train'].DEFAULT_OOD_FACTORS_LIST
        if args.mini or args.only_one_dshift:
            shifts = shifts[:1]
        callback_kwargs |= {
            'shift2loaders': {
                shift[0]: {n_shots: load_few_shot_datasets(ood_args, 'OOD_' + shift[0], n_shots) for n_shots in args.ood_log_nsshots} for shift in shifts
            },
            'ood_log_freq': args.ood_log_freq,
            'num_ood_epochs': args.num_ood_epochs,
            'ood_lr': args.ood_lr,
            'ood_unfreeze_subset': args.ood_unfreeze_subset,
        }
    return callback_kwargs
