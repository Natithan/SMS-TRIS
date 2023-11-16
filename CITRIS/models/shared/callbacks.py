import io
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

import wandb
from experiments.ood_utils import reset_model_and_optimizer, get_loader
from copy import deepcopy
import pytorch_lightning as pl
from experiments.datasets import Causal3DDataset, PinballDataset
from pytorch_lightning.callbacks import LearningRateMonitor
import matplotlib
matplotlib.use('Agg') # Need non-interactive backend for running exps in background
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import OrderedDict, defaultdict
from tqdm.auto import tqdm
from scipy.stats import spearmanr

import sys
sys.path.append('../../')
from models.shared.visualization import visualize_reconstruction, plot_target_assignment, plot_target_classification, \
    visualize_triplet_reconstruction, visualize_graph, plot_latents_mutual_information, visualize_frame_prediction, \
    visualize_triplet_reconstruction_parallel
from models.shared.utils import log_matrix, log_dict, evaluate_adj_matrix
from models.shared.causal_encoder import CausalEncoder
from models.shared.enco import ENCOGraphLearning
from util import tn, CUDAfy, Namespace


class ImageLogCallback(pl.Callback):
    """ Callback for creating visualizations for logging """

    def __init__(self, exmp_triplet_inputs, dataset, every_n_epochs=10, cluster=False, prefix='', exmp_fp_inputs=None, skip_frame_prediction=False):
        super().__init__()
        # self.imgs = exmp_inputs[0]
        self.imgs = exmp_triplet_inputs['imgs'] if exmp_triplet_inputs is not None else None
        # self.fp_imgs = exmp_fp_inputs[0] if exmp_fp_inputs is not None else None
        self.fp_imgs = exmp_fp_inputs['imgs'] if exmp_fp_inputs is not None else None
        # self.It1s = exmp_fp_inputs[1] if exmp_fp_inputs is not None else None
        self.It1s = exmp_fp_inputs['targets'] if exmp_fp_inputs is not None else None
        # if len(exmp_inputs) > 2 and len(exmp_inputs[1].shape) == len(self.imgs.shape):
        #     self.labels = exmp_inputs[1]
        #     self.extra_inputs = exmp_inputs[2:]
        # else:
        #     self.labels = self.imgs
        #     self.extra_inputs = exmp_inputs[1:]
        self.labels = exmp_triplet_inputs['imgs'] if exmp_triplet_inputs is not None else None
        self.extra_inputs = [exmp_triplet_inputs['targets']] if exmp_triplet_inputs is not None else []
        # if 'lat' in exmp_inputs:
        if exmp_triplet_inputs is not None and 'lat' in exmp_triplet_inputs:
            self.extra_inputs.append(exmp_triplet_inputs['lat'])
        self.dataset = dataset
        self.every_n_epochs = every_n_epochs
        self.prefix = prefix
        self.cluster = cluster
        self.skip_frame_prediction = skip_frame_prediction

    def on_train_epoch_end(self, trainer, pl_module):
        def log_fig(tag, fig):
            if fig is None:
                return
            # trainer.logger.experiment.add_figure(f'{self.prefix}{tag}', fig, global_step=trainer.global_step)
            # Nathan: adapted to wandb
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png')

            im = Image.open(img_buf)
            trainer.logger.log_image(f'{self.prefix}{tag}', [im])
            plt.close(fig)

        if hasattr(trainer.model, 'intv_classifier'):
            if (trainer.current_epoch+1) % (2 if not self.cluster else self.every_n_epochs) == 0:
                log_fig(f'target_classifier', plot_target_classification(trainer._results))


        # if hasattr(trainer.model, 'prior_t1'):
        #     if hasattr(trainer.model, 'prior_t1') and (trainer.current_epoch+1) % self.every_n_epochs == 0:
        #         log_fig('target_assignment', plot_target_assignment(trainer.model.prior_t1, dataset=self.dataset))
        #     if hasattr(trainer.model, 'last_target_assignment') and (trainer.current_epoch+1) % self.every_n_epochs == 0:
        #         log_fig('max_corr_target_assignment', plot_target_assignment(prior=None, dataset=self.dataset, target_assignment=trainer.model.last_target_assignment))
        if (trainer.current_epoch+1) % self.every_n_epochs == 0:
            if hasattr(trainer.model, 'get_target_assignment'):
                soft_target_assignment = trainer.model.get_target_assignment(hard=False)
                log_fig('target_assignment', plot_target_assignment(prior=None, dataset=self.dataset, target_assignment=soft_target_assignment))
            if hasattr(trainer.model, 'last_target_assignment'):
                log_fig('max_corr_target_assignment', plot_target_assignment(prior=None, dataset=self.dataset, target_assignment=trainer.model.last_target_assignment))
        if self.imgs is not None and (trainer.current_epoch+1) % self.every_n_epochs == 0:
            trainer.model.eval()
            images, labels = (el.to(trainer.model.device) for el in [self.imgs, self.labels])
            if len(images.shape) == 5:
                full_imgs, full_labels = images, labels
                images = images[:,0]
                labels = labels[:,0]
            else:
                full_imgs, full_labels = None, None

            # if trainer.model._get_name() != 'CITRISNF': # For CITRISNF, reconstruction is exactly the same as for its autoencoder because the flow is invertible
            from models.citris_nf import CITRISNF
            from models.shared import AutoregNormalizingFlow
            if trainer.model._get_name() not in [CITRISNF.__name__, AutoregNormalizingFlow.__name__]:
                for i in range(min(4, images.shape[0])):
                    log_fig(f'reconstruction_{i}', visualize_reconstruction(trainer.model, images[i], labels[i], self.dataset))
            
            # if hasattr(trainer.model, 'prior_t1'):
            if hasattr(trainer.model, 'triplet_prediction'):
                if full_imgs is not None:
                    # for i in range(min(4, full_imgs.shape[0])):
                    #     log_fig(f'triplet_visualization_{i}', visualize_triplet_reconstruction(trainer.model, full_imgs[i], full_labels[i], [e[i] for e in self.extra_inputs], dataset=self.dataset))

                    NUM_TRIPLET_SAMPLES = min(4, full_imgs.shape[0])
                    # log_fig(f'triplet_visualization',
                    #         visualize_triplet_reconstruction_parallel(
                    #             trainer.model,
                    #             full_imgs[:NUM_TRIPLET_SAMPLES],
                    #             full_labels[:NUM_TRIPLET_SAMPLES],
                    #             [e[:NUM_TRIPLET_SAMPLES] for e in self.extra_inputs], dataset=self.dataset)
                    #         )
                    for cheat_flag in (True, False):
                        log_fig(f'triplet_visualization{"_cheating" if cheat_flag else ""}',
                                visualize_triplet_reconstruction_parallel(
                                    trainer.model,
                                    full_imgs[:NUM_TRIPLET_SAMPLES],
                                    full_labels[:NUM_TRIPLET_SAMPLES],
                                    [e[:NUM_TRIPLET_SAMPLES] for e in self.extra_inputs],
                                    dataset=self.dataset,
                                    use_cheating_target_assignment=cheat_flag)
                                )
                    if not self.skip_frame_prediction:
                        NUM_SAMPLES = min(8, full_imgs.shape[0])
                        log_fig(f'Frame_prediction', visualize_frame_prediction(trainer.model, self.fp_imgs[:NUM_SAMPLES].to(trainer.model.device), self.It1s[:NUM_SAMPLES].to(trainer.model.device), dataset=self.dataset))

            trainer.model.train()


class CorrelationMetricsLogCallback(pl.Callback):
    """ Callback for extracting correlation metrics (R^2 and Spearman) """

    def __init__(self, dataset, every_n_epochs=10, num_train_epochs=100, cluster=False, test_dataset=None, mfp=False, ignore_learnt_psi=False):
        super().__init__()
        assert dataset is not None, "Dataset for correlation metrics cannot be None."
        self.dataset = dataset
        self.val_dataset = dataset
        self.test_dataset = test_dataset
        self.every_n_epochs = every_n_epochs
        self.num_train_epochs = num_train_epochs
        self.cluster = cluster
        self.log_postfix = ''
        self.extra_postfix = ''
        self.mfp = mfp
        self.ignore_learnt_psi = ignore_learnt_psi

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if isinstance(self.dataset, dict):
            dataset_dict = self.dataset
            data_len_sum = sum([len(dataset_dict[key]) for key in dataset_dict])
            for key in dataset_dict:
                self.dataset = dataset_dict[key]
                self.log_postfix = f'{self.extra_postfix}_{key}'
                self.test_model(trainer, pl_module)
            self.log_postfix = ''
            self.dataset = dataset_dict
        else:
            self.test_model(trainer, pl_module)

        results = trainer._results
        if 'validation_step.val_triplet_cnd' in results:
            triplet_loss_str = 'validation_step.val_triplet_cnd'
        else:
            triplet_loss_str = 'validation_step.val_loss'
        if triplet_loss_str not in results: # Nathan
            triplet_loss_str += ".0" # When there are multiple validation loaders
        if triplet_loss_str in results:
            val_comb_loss = results[triplet_loss_str].value / results[triplet_loss_str].cumulated_batch_size
            new_val_dict = {'triplet_loss': val_comb_loss}
            for key in ['on_validation_epoch_end.corr_callback_r2_matrix_diag',
                        'on_validation_epoch_end.corr_callback_spearman_matrix_diag',
                        'on_validation_epoch_end.corr_callback_r2_matrix_max_off_diag',
                        'on_validation_epoch_end.corr_callback_spearman_matrix_max_off_diag']:
                if key in results:
                    val = results[key].value
                    new_val_dict[key.split('_',5)[-1]] = val
                    if key.endswith('matrix_diag'):
                        val = 1 - val
                    val_comb_loss += val
            pl_module.log(f'val_comb_loss{self.log_postfix}{self.extra_postfix}', val_comb_loss)
            new_val_dict = {key: (val.item() if isinstance(val, torch.Tensor) else val) for key, val in new_val_dict.items()}
            if self.cluster:
                s = f'[Epoch {trainer.current_epoch}] ' + ', '.join([f'{key}: {new_val_dict[key]:5.3f}' for key in sorted(list(new_val_dict.keys()))])
                print(s)


    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_dataset is None:
            print('Skipping correlation metrics testing due to missing dataset...')
        else:
            val_dataset = self.dataset
            self.dataset = self.test_dataset
            self.log_postfix = '_test'
            self.extra_postfix = '_test'
            self.on_validation_epoch_end(trainer, pl_module)
            self.dataset = val_dataset
            self.log_postfix = ''
            self.extra_postfix = ''

    @torch.no_grad()
    def test_model(self, trainer, pl_module):
        is_training = pl_module.training
        pl_module = pl_module.eval()
        # region Encode whole dataset with pl_module. all_encs = z, all_lats = c
        loader = data.DataLoader(self.dataset, batch_size=256, drop_last=False, shuffle=False)
        all_encs, all_latents = [], []
        for batch in loader:
            # inps, *_, latents = batch
            inps, latents = batch['img'], batch['lat']
            if self.mfp:
                encs, _ = pl_module.encode(latents.to(pl_module.device), from_c=True) # c -> d -> e
            else:
                encs = pl_module.encode(inps.to(pl_module.device)).cpu() # x or y -> z
            all_encs.append(encs)
            all_latents.append(latents)
        all_encs = torch.cat(all_encs, dim=0)
        all_latents = torch.cat(all_latents, dim=0)
        # endregion
        # Normalize z for stable gradient signals
        all_encs = (all_encs - all_encs.mean(dim=0, keepdim=True)) / all_encs.std(dim=0, keepdim=True).clamp(min=1e-2)
        # region Create new tensor dataset for training (50%) and testing (50%)
        full_dataset = data.TensorDataset(all_encs, all_latents)
        train_size = int(0.5 * all_encs.shape[0])
        test_size = all_encs.shape[0] - train_size
        train_dataset, test_dataset = data.random_split(full_dataset, 
                                                        lengths=[train_size, test_size], 
                                                        generator=torch.Generator().manual_seed(42))
        # endregion
        # Train network to predict causal factors (c) from latent variables (z)
        target_assignment = self.set_target_assignment(all_encs, pl_module)
        encoder = self.train_network(pl_module, train_dataset, target_assignment)
        encoder.eval()
        # Record predictions of model on test and calculate distances
        test_inps, test_labels = all_encs[test_dataset.indices], all_latents[test_dataset.indices]
        test_exp_inps, test_exp_labels = expand_per_group(test_inps.cpu(), target_assignment.cpu(), test_labels, flatten_inp=False)
        pred_dict = encoder.forward(test_exp_inps.to(pl_module.device))
        for key in pred_dict:
            pred_dict[key] = pred_dict[key].cpu()
        _, dists, norm_dists = encoder.calculate_loss_distance(pred_dict, test_exp_labels)
        # Calculate statistics (R^2, pearson, etc.)
        avg_norm_dists, r2_matrix = self.log_R2_statistic(trainer, encoder, pred_dict, test_labels, norm_dists, pl_module=pl_module)
        self.log_Spearman_statistics(trainer, encoder, pred_dict, test_labels, pl_module=pl_module)
        if is_training:
            # pl_module = pl_module.train()
            pl_module.train()
        return r2_matrix

    def set_target_assignment(self, all_encs, pl_module):
        # if hasattr(pl_module, 'prior_t1') and (not self.ignore_prior_t1):
        #     target_assignment = pl_module.prior_t1.get_target_assignment(hard=True)
        # elif hasattr(pl_module, 'target_assignment') and pl_module.target_assignment is not None:
        #     target_assignment = pl_module.target_assignment.clone()
        if not self.ignore_learnt_psi:
            target_assignment = pl_module.get_target_assignment(hard=True)
        else:
            if hasattr(pl_module, 'target_assignment') and pl_module.target_assignment is not None:
                target_assignment = pl_module.target_assignment.clone()
            else:
                target_assignment = torch.eye(all_encs.shape[-1])
        return target_assignment

    def train_network(self, pl_module, train_dataset, target_assignment):
        device = pl_module.device
        causal_var_info = self.get_causal_var_info(pl_module)
        # We use one, sufficiently large network that predicts for any input all causal variables
        # To iterate over the different sets, we use a mask which is an extra input to the model
        # This is more efficient than using N networks and showed same results with large hidden size

        # Paper:  "Since in our setup, multiple latent
        # variables can jointly describe a single causal variable, we first learn a mapping between
        # such, e.g., with an MLP. For CITRIS, we apply one MLP per set of latent variables that
        # are assigned to the same causal factor by ψ. The MLP is then trained to predict all causal
        # factors per set of latents, on which we measure the correlation. Thereby, no gradients are
        # propagated through the model."
        # iVAE∗ and SlowVAE do not learn an assignment of latent to causal factors. As an alternative, we assign each latent dimension to the causal factor it has the highest correlation with. Although this gives the baselines a considerable advantage, it shows whether CITRIS can improve upon the baselines beyond finding a good latent to causal factor assignment.

        # This the MLP described above
        encoder = CausalEncoder(c_hid=128,
                                lr=4e-3,
                                causal_var_info=causal_var_info,
                                single_linear=True,
                                c_in=pl_module.hparams.num_latents*2 if not self.mfp else pl_module.num_latents*2,
                                warmup=0)
        optimizer = encoder.get_main_optimizer()

        train_loader = data.DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=512)
        target_assignment = target_assignment.to(device)
        encoder.to(device)
        encoder.train()
        with torch.enable_grad():
            range_iter = range(self.num_train_epochs)
            if not self.cluster:
                range_iter = tqdm(range_iter, leave=False, desc=f'Training correlation encoder {self.log_postfix}')
            for epoch_idx in range_iter:
                avg_loss = 0.0
                for inps, latents in train_loader:
                    inps = inps.to(device)
                    latents = latents.to(device)
                    inps, latents = expand_per_group(inps, target_assignment, latents)
                    loss = encoder._get_loss([inps, latents], mode=None)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item()
        return encoder

    def get_causalenc_optimizer(self, encoder):
        optimizer, _ = encoder.configure_optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        return optimizer

    def get_causal_var_info(self, pl_module):
        if hasattr(pl_module, 'causal_encoder'):
            causal_var_info = pl_module.causal_encoder.hparams.causal_var_info
        else:
            causal_var_info = pl_module.hparams.causal_var_info
        return causal_var_info

    def log_R2_statistic(self, trainer, encoder, pred_dict, test_labels, norm_dists, pl_module=None):
        avg_pred_dict = OrderedDict()
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            gt_vals = test_labels[...,i]
            if var_info.startswith('continuous'):
                avg_pred_dict[var_key] = gt_vals.mean(dim=0, keepdim=True).expand(gt_vals.shape[0],)
            elif var_info.startswith('angle'):
                avg_angle = torch.atan2(torch.sin(gt_vals).mean(dim=0, keepdim=True), 
                                        torch.cos(gt_vals).mean(dim=0, keepdim=True)).expand(gt_vals.shape[0],)
                avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2*np.pi, avg_angle)
                avg_pred_dict[var_key] = torch.stack([torch.sin(avg_angle), torch.cos(avg_angle)], dim=-1)
            elif var_info.startswith('categ'):
                gt_vals = gt_vals.long()
                mode = torch.mode(gt_vals, dim=0, keepdim=True).values
                avg_pred_dict[var_key] = F.one_hot(mode, int(var_info.split('_')[-1])).float().expand(gt_vals.shape[0], -1)
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in R2 statistics.'
        _, _, avg_norm_dists = encoder.calculate_loss_distance(avg_pred_dict, test_labels, keep_sign=True)

        r2_matrix = []
        for var_key in encoder.hparams.causal_var_info:
            ss_res = (norm_dists[var_key] ** 2).mean(dim=0)
            ss_tot = (avg_norm_dists[var_key] ** 2).mean(dim=0, keepdim=True)
            r2 = 1 - ss_res / ss_tot
            r2_matrix.append(r2)
        r2_matrix = torch.stack(r2_matrix, dim=-1).cpu().numpy()
        self.log_corr_stuff(r2_matrix, 'r2', 'R^2', encoder, pl_module, trainer)
        return avg_norm_dists, r2_matrix

    def log_corr_stuff(self, corr_matrix, corr_name, corr_name_fancy, encoder, pl_module, trainer):
        if '_all_latents' not in self.log_postfix: # Nathan: decided not to log all latents: their only use is allowing grouped_latents to be used, and they just clutter the log
            log_matrix(corr_matrix, trainer, f'{corr_name}_matrix' + self.log_postfix)
            self._log_heatmap(trainer=trainer,
                              values=corr_matrix,
                              tag=f'{corr_name}_matrix',
                              title=f'{corr_name_fancy} Matrix',
                              xticks=[key for key in encoder.hparams.causal_var_info],
                              pl_module=pl_module)

    def log_pearson_statistic(self, trainer, encoder, pred_dict, test_labels, norm_dists, avg_gt_norm_dists, pl_module=None):
        avg_pred_dict = OrderedDict()
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            pred_vals = pred_dict[var_key]
            if var_info.startswith('continuous'):
                pred_vals = pred_vals.squeeze(dim=-1)
                avg_pred_dict[var_key] = pred_vals.mean(dim=0, keepdim=True).expand(pred_vals.shape[0], -1)
            elif var_info.startswith('angle'):
                angles = torch.atan(pred_vals[...,0] / pred_vals[...,1])
                avg_angle = torch.atan2(torch.sin(angles).mean(dim=0, keepdim=True), 
                                        torch.cos(angles).mean(dim=0, keepdim=True)).expand(pred_vals.shape[0], -1)
                avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2*np.pi, avg_angle)
                avg_pred_dict[var_key] = avg_angle
            elif var_info.startswith('categ'):
                pred_vals = pred_vals.argmax(dim=-1)
                mode = torch.mode(pred_vals, dim=0, keepdim=True).values
                avg_pred_dict[var_key] = mode.expand(pred_vals.shape[0], -1)
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in Pearson statistics.'
        _, _, avg_pred_norm_dists = encoder.calculate_loss_distance(pred_dict, gt_vec=torch.stack([avg_pred_dict[key] for key in avg_pred_dict], dim=-1), keep_sign=True)

        pearson_matrix = []
        for var_key in encoder.hparams.causal_var_info:
            var_info = encoder.hparams.causal_var_info[var_key]
            pred_dist, gt_dist = avg_pred_norm_dists[var_key], avg_gt_norm_dists[var_key]
            nomin = (pred_dist * gt_dist[:,None]).sum(dim=0)
            denom = torch.sqrt((pred_dist**2).sum(dim=0) * (gt_dist[:,None]**2).sum(dim=0))
            p = nomin / denom.clamp(min=1e-5)
            pearson_matrix.append(p)
        pearson_matrix = torch.stack(pearson_matrix, dim=-1).cpu().numpy()
        # log_matrix(pearson_matrix, trainer, 'pearson_matrix' + self.log_postfix)
        # self._log_heatmap(trainer=trainer,
        #                   values=pearson_matrix,
        #                   tag='pearson_matrix',
        #                   title='Pearson Matrix',
        #                   xticks=[key for key in encoder.hparams.causal_var_info],
        #                   pl_module=pl_module)
        self.log_corr_stuff(pearson_matrix, 'pearson', 'Pearson', encoder, pl_module, trainer)

    def log_Spearman_statistics(self, trainer, encoder, pred_dict, test_labels, pl_module=None):
        spearman_matrix = []
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            gt_vals = test_labels[...,i]
            pred_val = pred_dict[var_key]
            if var_info.startswith('continuous'):
                spearman_preds = pred_val.squeeze(dim=-1)  # Nothing needs to be adjusted
            elif var_info.startswith('angle'):
                spearman_preds = F.normalize(pred_val, p=2.0, dim=-1)
                gt_vals = torch.stack([torch.sin(gt_vals), torch.cos(gt_vals)], dim=-1)
            elif var_info.startswith('categ'):
                spearman_preds = pred_val.argmax(dim=-1).float()
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in Spearman statistics.'

            gt_vals = gt_vals.cpu().numpy()
            spearman_preds = spearman_preds.cpu().numpy()
            results = torch.zeros(spearman_preds.shape[1],)
            for j in range(spearman_preds.shape[1]):
                if len(spearman_preds.shape) == 2:
                    if np.unique(spearman_preds[:,j]).shape[0] == 1:
                        results[j] = 0.0
                    else:
                        results[j] = spearmanr(spearman_preds[:,j], gt_vals).correlation
                elif len(spearman_preds.shape) == 3:
                    num_dims = spearman_preds.shape[-1]
                    for k in range(num_dims):
                        if np.unique(spearman_preds[:,j,k]).shape[0] == 1:
                            results[j] = 0.0
                        else:
                            results[j] += spearmanr(spearman_preds[:,j,k], gt_vals[...,k]).correlation
                    results[j] /= num_dims
                
            spearman_matrix.append(results)
        
        spearman_matrix = torch.stack(spearman_matrix, dim=-1).cpu().numpy()
        # log_matrix(spearman_matrix, trainer, 'spearman_matrix' + self.log_postfix)
        # self._log_heatmap(trainer=trainer,
        #                   values=spearman_matrix,
        #                   tag='spearman_matrix',
        #                   title='Spearman\'s Rank Correlation Matrix',
        #                   xticks=[key for key in encoder.hparams.causal_var_info],
        #                   pl_module=pl_module)
        self.log_corr_stuff(spearman_matrix, 'spearman', 'Spearman\'s Rank Correlation', encoder, pl_module, trainer)

    def _log_heatmap(self, trainer, values, tag, title=None, xticks=None, yticks=None, xlabel=None, ylabel=None, pl_module=None):
        if ylabel is None:
            ylabel = 'Target dimension'
        if xlabel is None:
            xlabel = 'True causal variable'
        if yticks is None:
            yticks = self.dataset.target_names()+['No variable']
            if values.shape[0] > len(yticks):
                yticks = [f'Dim {i+1}' for i in range(values.shape[0])]
            if len(yticks) > values.shape[0]:
                yticks = yticks[:values.shape[0]]
        if xticks is None:
            xticks = self.dataset.target_names()
        fig = plt.figure(figsize=(min(6, values.shape[1]/1.25), min(6, values.shape[0]/1.25)))
        sns.heatmap(values, annot=True,
                    yticklabels=yticks,
                    xticklabels=xticks,
                    fmt='3.2f')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        plt.tight_layout()

        # trainer.logger.experiment.add_figure(tag + self.log_postfix, fig, global_step=trainer.global_step)
        # Nathan: adapted to wandb
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')

        im = Image.open(img_buf)
        trainer.logger.log_image(tag + self.log_postfix, [im])
        plt.close(fig)

        if values.shape[0] == values.shape[1] + 1:  # Remove 'lambda_sparse' variables
            values = values[:-1]

        if values.shape[0] == values.shape[1]:
            avg_diag = np.diag(values).mean()
            max_off_diag = (values - np.eye(values.shape[0]) * 10).max(axis=-1).mean()
        else: # Nathan
            # f2c = self.dataset.FINE2COARSE
            # fv = self.dataset.FACTORS
            # iv = self.dataset.INTERVENED_FACTORS
            # col_diag_idxs = list(range(len(fv)))
            # row_diag_idxs = []
            # for i, true_causal_var in enumerate(fv):
            #     coarse_var = f2c[true_causal_var]
            #     if coarse_var in iv:
            #         row_diag_idxs.append(iv.index(coarse_var))
            #     else:
            #         row_diag_idxs.append(len(iv))
            # avg_diag = np.mean([values[i,j] for i,j in zip(row_diag_idxs, col_diag_idxs)])
            # m = values.copy()
            # m[row_diag_idxs, col_diag_idxs] = -100
            # max_off_diag = m.max(axis=0).mean()
            avg_diag, max_off_diag = self.dataset.get_coarse2fine_correlation_summary(values)
        if pl_module is None:
            trainer.logger.experiment.add_scalar(f'corr_callback_{tag}_diag{self.log_postfix}', avg_diag, global_step=trainer.global_step)
            trainer.logger.experiment.add_scalar(f'corr_callback_{tag}_max_off_diag{self.log_postfix}', max_off_diag, global_step=trainer.global_step)
        else:
            pl_module.log(f'corr_callback_{tag}_diag{self.log_postfix}', avg_diag)
            pl_module.log(f'corr_callback_{tag}_max_off_diag{self.log_postfix}', max_off_diag)


def expand_per_group(inps, target_assignment, latents, flatten_inp=True, detach=True):
    if detach:
        target_assignment = target_assignment.detach()
    ta = target_assignment[None,:,:].expand(inps.shape[0], -1, -1)
    inps = torch.cat([inps[:,:,None] * ta, ta], dim=-2).permute(0, 2, 1)
    latents = latents[:,None].expand(-1, inps.shape[1], -1)
    if flatten_inp:
        inps = inps.flatten(0, 1)
        latents = latents.flatten(0, 1)
    return inps, latents


class GraphLogCallback(pl.Callback):
    """ Callback for creating visualizations for logging """

    def __init__(self, every_n_epochs=1, dataset=None, cluster=False):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        if dataset is not None and hasattr(dataset, 'get_adj_matrix'):
            self.gt_adj_matrix = dataset.get_adj_matrix()
        else:
            self.gt_adj_matrix = None
        self.last_adj_matrix = None
        self.cluster = cluster
        self.log_string = None

    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(trainer.model, 'prior_t1') and hasattr(trainer.model.prior_t1, 'get_adj_matrix'):
            adj_matrix = trainer.model.prior_t1.get_adj_matrix(hard=True).cpu().detach()
            log_matrix(adj_matrix.numpy(), trainer, 'instantaneous_adjacency_matrix')
            if (trainer.current_epoch+1) % self.every_n_epochs == 0:
                if self.last_adj_matrix is None or (self.last_adj_matrix != adj_matrix).any():
                    # Don't visualize the same graph several times, reduces tensorboard size
                    self.last_adj_matrix = adj_matrix
                    if hasattr(trainer.model.hparams, 'var_names'):
                        var_names = trainer.model.hparams.var_names
                    else:
                        var_names = []
                    while len(var_names) < adj_matrix.shape[0]-1:
                        var_names.append(f'C{len(var_names)}')
                    if len(var_names) == adj_matrix.shape[0]-1:
                        var_names.append('Noise')

                    fig = visualize_graph(nodes=var_names, adj_matrix=adj_matrix)
                    trainer.logger.experiment.add_figure('instantaneous_graph', fig, global_step=trainer.global_step)
                    plt.close(fig)

            if self.gt_adj_matrix is not None:
                metrics = evaluate_adj_matrix(adj_matrix, self.gt_adj_matrix)
                log_dict(metrics, 'instantaneous_adjacency_matrix_metrics', trainer=trainer)
                for key in metrics:
                    trainer.logger.experiment.add_scalar(f'adj_matrix_{key}', metrics[key], global_step=trainer.global_step)
                self.log_string = f'[Epoch {trainer.current_epoch+1}] ' + ', '.join([f'{key}: {metrics[key]:4.2f}' for key in sorted(list(metrics.keys()))])

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.cluster and self.log_string is not None:
            print(self.log_string)
            self.log_string = None
    
    def on_test_epoch_start(self, trainer, pl_module):
        if hasattr(trainer.model, 'prior_t1') and hasattr(trainer.model.prior_t1, 'get_adj_matrix'):
            adj_matrix = trainer.model.prior_t1.get_adj_matrix(hard=True).cpu().detach()
            if self.gt_adj_matrix is not None:
                metrics = evaluate_adj_matrix(adj_matrix, self.gt_adj_matrix)
                for key in metrics:
                    pl_module.log(f'test_adj_matrix_{key}', torch.FloatTensor([metrics[key]]))


class SparsifyingGraphCallback(pl.Callback):
    """ Callback for creating visualizations for logging """

    def __init__(self, dataset, lambda_sparse=[0.02], cluster=False, prefix='',mini=False):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.cluster = cluster
        self.dataset = dataset
        self.prefix = prefix
        self.gt_adj_matrix = None
        self.gt_temporal_adj_matrix = None
        if dataset is not None and hasattr(dataset, 'get_adj_matrix'):
            self.gt_adj_matrix = dataset.get_adj_matrix()
        if dataset is not None and hasattr(dataset, 'get_temporal_adj_matrix'):
            self.gt_temporal_adj_matrix = dataset.get_temporal_adj_matrix()
        self.mini = mini

    def set_test_prefix(self, prefix):
        self.prefix = '_' + prefix

    def set_test_module_ckpt_path(self, module_ckpt_path): # for storing a reference to the pl_module in the enco_net.pkl file, b/c the raw pl_module isn't pickleable
        self.module_ckpt_path = module_ckpt_path
        
    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        pl_module = pl_module.eval()
        self.dataset.seq_len = 2
        logs = {}
        # log_filename = os.path.join(trainer.logger.log_dir, f'enco_adj_matrices{self.prefix}.npz')
        log_filename = os.path.join(trainer.logger.experiment.dir, f'enco_adj_matrices{self.prefix}.npz')  # Nathan: logger.experiment.dir instead of logger.log_dir for wandb logger
        lambda_sparse = self.lambda_sparse
        if not self.cluster:
            lambda_sparse = tqdm(lambda_sparse, desc='ENCO sparsity settings', leave=False)
        for l_sparse in lambda_sparse:
            enco = ENCOGraphLearning(model=pl_module,
                                     verbose=not self.cluster,
                                     lambda_sparse=l_sparse,
                                     debug=self.mini)
            temporal_adj_matrix, instantaneous_adj_matrix = enco.learn_graph(self.dataset)
            if self.gt_adj_matrix is not None:
                metrics = evaluate_adj_matrix(instantaneous_adj_matrix, self.gt_adj_matrix)
                for key in metrics:
                    pl_module.log(f'test{self.prefix}_instant_graph_{key}_lambda_sparse_{l_sparse}', torch.Tensor([metrics[key]]))
                    logs[f'{l_sparse}_instantaneous_{key}'] = np.array([metrics[key]])
            if self.gt_temporal_adj_matrix is not None:
                metrics_temporal = evaluate_adj_matrix(temporal_adj_matrix, self.gt_temporal_adj_matrix)
                for key in metrics_temporal:
                    pl_module.log(f'test{self.prefix}_temporal_graph_{key}_lambda_sparse_{l_sparse}', torch.Tensor([metrics_temporal[key]]))
                    logs[f'{l_sparse}_temporal_{key}'] = np.array([metrics_temporal[key]])
            logs[f'{l_sparse}_temporal_adj_matrix'] = temporal_adj_matrix
            logs[f'{l_sparse}_instantaneous_adj_matrix'] = instantaneous_adj_matrix
            np.savez_compressed(log_filename, **logs)
            visualize_graph(nodes=None, adj_matrix=temporal_adj_matrix)
            plt.savefig(os.path.join(trainer.logger.experiment.dir, f'temporal_graph_lsparse_{str(l_sparse).replace(".", "_")}{self.prefix}.pdf'))
            plt.close() # Nathan changed logger.log_dir to logger.experiment.dir
            if instantaneous_adj_matrix is not None:
                visualize_graph(nodes=None, adj_matrix=instantaneous_adj_matrix)
                plt.savefig(os.path.join(trainer.logger.experiment.dir, f'instantaneous_graph_lsparse_{str(l_sparse).replace(".", "_")}{self.prefix}.pdf')) # Nathan changed logger.log_dir to logger.experiment.dir
                plt.close()

            # Nathan
            if not pl_module.hparams['fullview_baseline']:
                fp = os.path.join(trainer.default_root_dir, 'enco_net.pkl')
                to_pkl = {"pl_module_path": self.module_ckpt_path,
                          "enco_prior": enco.net,
                          "temp_adj_matrix": temporal_adj_matrix, "inst_adj_matrix": instantaneous_adj_matrix}
                with open(fp, 'wb') as f:
                    pickle.dump(to_pkl, f)


class BaselineCorrelationMetricsLogCallback(CorrelationMetricsLogCallback):
    """
    Adapting the correlation metrics callback to the baselines by first running
    the correlation estimation for every single latent variable, and then grouping
    them according to the highest correlation with a causal variable.
    """

    def __init__(self, *args, ignore_learnt_psi=True, **kwargs):
        super().__init__(*args, ignore_learnt_psi=ignore_learnt_psi,**kwargs)
        if self.test_dataset is None:
            self.test_dataset = self.val_dataset

    @torch.no_grad()
    def on_validation_epoch_start(self, trainer, pl_module, is_test=False):
        self.log_postfix = '_all_latents' + ('_test' if is_test else '')

        pl_module.target_assignment = None
        r2_matrix = self.test_model(trainer, pl_module)
        max_r2 = torch.from_numpy(r2_matrix).argmax(dim=-1)
        ta = F.one_hot(max_r2, num_classes=r2_matrix.shape[-1]).float()
        if isinstance(self.dataset, Causal3DDataset) and self.dataset.coarse_vars:
            ta = torch.cat([ta[:,:3].sum(dim=-1, keepdims=True), ta[:,3:5].sum(dim=-1, keepdims=True), ta[:,5:]], dim=-1)
        elif isinstance(self.dataset, PinballDataset):
            ta = torch.cat([ta[:,:4].sum(dim=-1, keepdims=True),
                            ta[:,4:9].sum(dim=-1, keepdims=True),
                            ta[:,9:]], dim=-1)
        pl_module.target_assignment = ta
        assert ta.shape == pl_module.last_target_assignment.shape
        pl_module.last_target_assignment.data = ta # for enco

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module, is_test=False):
        self.log_postfix = '_grouped_latents' + ('_test' if is_test else '')
        self.test_model(trainer, pl_module)

        if not is_test:
            results = trainer._results
            if any([f'validation_step.val_triplet_cnd{sf}' in results for sf in ['', '_cheating']]):
                triplet_loss_str = 'validation_step.val_triplet_cnd_cheating'
            else:
                triplet_loss_str = 'validation_step.val_loss' # TODO figure out why for pong we get here
            if triplet_loss_str in results:
                val_comb_loss = results[triplet_loss_str].value / results[triplet_loss_str].cumulated_batch_size
                new_val_dict = {'triplet_loss': val_comb_loss}
                for key in ['on_validation_epoch_end.corr_callback_r2_matrix_diag_grouped_latents',
                            'on_validation_epoch_end.corr_callback_spearman_matrix_diag_grouped_latents',
                            'on_validation_epoch_end.corr_callback_r2_matrix_max_off_diag_grouped_latents',
                            'on_validation_epoch_end.corr_callback_spearman_matrix_max_off_diag_grouped_latents']:
                    if key in results:
                        val = results[key].value
                        new_val_dict[key.split('_',5)[-1]] = val
                        if 'matrix_diag' in key:
                            val = 1 - val
                        val_comb_loss += val
                pl_module.log(f'val_comb_loss{self.log_postfix}{self.extra_postfix}', val_comb_loss)
                new_val_dict = {key: (val.item() if isinstance(val, torch.Tensor) else val) for key, val in new_val_dict.items()}
                if self.cluster:
                    s = f'[Epoch {trainer.current_epoch}] ' + ', '.join([f'{key}: {new_val_dict[key]:5.3f}' for key in sorted(list(new_val_dict.keys()))])
                    print(s)

    @torch.no_grad()
    def on_test_epoch_start(self, trainer, pl_module):
        self.dataset = self.test_dataset
        self.on_validation_epoch_start(trainer, pl_module, is_test=True)

    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module):
        self.on_validation_epoch_end(trainer, pl_module, is_test=True)
        self.dataset = self.val_dataset


class ManualCheckpointCallback(pl.Callback):
    """
    Callback to manually save the model.
    """

    def __init__(self, epochs_to_save_at, ckpt_dir):
        super().__init__()
        self.epochs_to_save_at = epochs_to_save_at
        self.ckpt_dir = ckpt_dir

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch in self.epochs_to_save_at:
            ckpt_path = os.path.join(self.ckpt_dir, f'manual_epoch_{trainer.current_epoch}.ckpt')
            trainer.save_checkpoint(ckpt_path)


class FramePredictionCallback(pl.Callback):

    def __init__(self, dataloader, mini, every_n_epochs=1, pt_args=None, **kwargs):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.dataloader = dataloader
        self.mini = mini
        self.kwargs = kwargs
        self.c_factors = dataloader.dataset.SHORT_FACTORS
        self.pt_args = pt_args
        self.get_losses_args = Namespace(**{
            "use_mean_to_autoregress_str": self.pt_args.use_mean_to_autoregress_str,
        })

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.log_nfp_results(trainer, pl_module)

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     self.log_nfp_results(trainer, pl_module)

    def log_nfp_results(self, trainer, pl_module):
        # avg_mse_x_mean, avg_normed_dist_c_mean, avg_normed_dist_per_c = None, None, None
        match self.pt_args.use_mean_to_autoregress_str:
            case 'both':
                postfixes = ['_mar', '_sar']
            case 'True':
                postfixes = ['_mar']
            case 'False':
                postfixes = ['_sar']
            case _:
                raise ValueError(f'Unknown use_mean_to_autoregress_str: {self.pt_args.use_mean_to_autoregress_str}')
        avg_metrics = {}
        for postfix in postfixes:
            avg_metrics.update({
                f'mse_x_mean{postfix}': None,
                f'normed_dist_c_mean{postfix}': None,
                f'normed_dist_per_c{postfix}': None
            })
        # avg_metrics = {
        #     'mse_x_mean': None, # the x_mse is actually already being logged during validation in log_mse_losses. Keeping here for sanity check. Later update: the sanity check revealed a difference due to within-autoregression sampling or not
        #     'normed_dist_c_mean': None,
        #     'normed_dist_per_c': None
        # }
        count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                batch = CUDAfy(batch)
                loss_dict = pl_module.get_losses(batch, args=self.get_losses_args)
                batch_size = list(batch.values())[0].shape[0]
                count += batch_size
                for key in avg_metrics:
                    if avg_metrics[key] is None:
                        avg_metrics[key] = tn(loss_dict[key]) * batch_size
                    else:
                        avg_metrics[key] += tn(loss_dict[key]) * batch_size
                if self.mini and batch_idx == 1:
                    break
        for key, val in avg_metrics.items():
            val /= count
            if 'per_c' in key:
                if batch_idx == 0:
                    assert len(val) == len(self.c_factors)
                for subval, factor in zip(val, self.c_factors):
                    pl_module.log(f'per_cdim/val_nfp_{key}_{factor}', subval)
            else:
                pl_module.log(f'val_nfp_{key}', val)


class FewShotOODCallback(pl.Callback):

    def __init__(self, shift2loaders, every_n_epochs=1, num_ood_epochs=50, ood_lr=1e-3, ood_unfreeze_subset='uf_all', pt_args=None, **kwargs):
        self.every_n_epochs = every_n_epochs
        self.shift2loaders = shift2loaders
        self.num_ood_epochs = num_ood_epochs
        self.ood_lr = ood_lr
        self.ood_unfreeze_subset = ood_unfreeze_subset
        self.kwargs = kwargs
        self.pt_args = pt_args
        # maybe_wrapped_dataset = list(list(self.shift2loaders.values())[0].values())[0].dataset 1 extra layer to unpack
        maybe_wrapped_dataset = list(list(list(self.shift2loaders.values())[0].values())[0].values())[0].dataset
        if hasattr(maybe_wrapped_dataset, 'dataset'):
            unwrapped_dataset = maybe_wrapped_dataset.dataset
        else:
            unwrapped_dataset = maybe_wrapped_dataset
        self.data_class = unwrapped_dataset.__class__
        self.id_train_loader = None
        self.ood_ns_shots = pt_args.ood_log_nsshots

    def set_id_train_loader(self, trainer, pl_module):
        loader_args = Namespace(**{
            'model_classes': [pl_module.__class__],
            'num_workers': self.pt_args.num_workers,
            'batch_size': self.pt_args.batch_size,
            'test_batch_size': self.pt_args.batch_size,
        })
        self.id_train_loader = get_loader(loader_args, self.data_class, data_folder=self.pt_args.data_dir, split='train', mini=self.pt_args.mini)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_ood_results(pl_module, trainer)

    # def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     self.log_ood_results(pl_module, trainer)

    def log_ood_results(self, pl_module, trainer, log_with_wandb=False):
        from experiments.domain_adapt import train_da, ttest_da
        if self.id_train_loader is None:
            self.set_id_train_loader(trainer, pl_module)
        if trainer.current_epoch % self.every_n_epochs == 0:
            # save model to temporary checkpoint, with process id in name to avoid overwriting
            ckpt_path = os.path.join(trainer.default_root_dir, f'temp_ood_{os.getpid()}.ckpt')
            trainer.save_checkpoint(ckpt_path)
            # mean_cnd = None
            nshots2sum_cnd = {k: None for k in self.ood_ns_shots}
            for shift, nshots2splits in self.shift2loaders.items():
                for ood_n_shots, split2loader in nshots2splits.items():
                    reset_args = Namespace(**{
                        'data_class': self.data_class,
                        'beta_coral': None,
                        'fixed_logstd': True,
                        'sample_mean': True,
                        'coral': False,
                        'num_epochs': self.num_ood_epochs,
                        'lr': self.ood_lr,
                        'log_logstd_hists': False,
                        "use_mean_to_autoregress_str": self.pt_args.use_mean_to_autoregress_str,
                    })
                    model, optimizer, scheduler_dict = reset_model_and_optimizer(ckpt_path, pl_module.__class__,
                                                                                 args=reset_args,
                                                                                 unfreeze_subset=self.ood_unfreeze_subset,
                                                                                 ood_shifts=[shift],
                                                                                 loaders=split2loader,
                                                                                 id_train_loader=self.id_train_loader)
                    train_da_args = Namespace(**{
                        "use_maxcorr_psi": None,
                        "wandb_ids": [],
                        "reset_matching": False,
                        "num_epochs": self.num_ood_epochs,
                        "check_val_every_n_epoch": self.num_ood_epochs // 10,
                        "mini": self.pt_args.mini,
                        "coral": False,
                        'sample_mean': False,
                        "use_mean_to_autoregress_str": self.pt_args.use_mean_to_autoregress_str,
                        'per_batch_logging': self.pt_args.per_batch_logging,
                        'log_all_train_values': self.pt_args.log_all_train_values,
                    })
                    ttest_da_args = deepcopy(train_da_args)
                    if ood_n_shots > 0:
                        dct = train_da(train_da_args, model, ood_n_shots, optimizer, scheduler_dict,
                                       split2loader['train'], split2loader['val'],
                                       unfreeze_setting=self.ood_unfreeze_subset, distribution=f'OOD_{shift}')
                        model_for_ttest = dct['best_val']['model']
                        best_val_epoch = dct['best_val']['epoch']
                        pl_module.log(f'per_distr_shift/ood_{ood_n_shots}sh_{shift}_best_val_epoch', best_val_epoch)
                        if log_with_wandb:
                            wandb.log({f'per_distr_shift/ood_{ood_n_shots}sh_{shift}_best_val_epoch': best_val_epoch})
                    else:
                        model_for_ttest = model
                    loss_dict = ttest_da(ttest_da_args, model_for_ttest, split2loader['test'],
                                         distribution=f'OOD_{shift}')
                    # cnd_loss_mean = loss_dict['normed_dist_c_mean']
                    # cnd_loss_per_c = loss_dict['normed_dist_per_c']
                    cnd_losses_mean = {k:el for k,el in loss_dict.items() if 'normed_dist_c_mean' in k}
                    cnd_losses_per_c = {k:el for k,el in loss_dict.items() if 'normed_dist_per_c' in k}
                    if nshots2sum_cnd[ood_n_shots] is None:
                        nshots2sum_cnd[ood_n_shots] = cnd_losses_mean
                    else:
                        # nshots2sum_cnd[ood_n_shots] += cnd_losses_per_c
                        for k,v in cnd_losses_mean.items():
                            nshots2sum_cnd[ood_n_shots][k] += v
                    # log at current epoch
                    # pl_module.log(f'per_distr_shift/ood_{ood_n_shots}sh_{shift}_cnd_loss', cnd_loss_mean.item())
                    for k,v in cnd_losses_mean.items():
                        postfix = k.split('normed_dist_c_mean')[1]
                        pl_module.log(f'per_distr_shift/ood_{ood_n_shots}sh_{shift}_cnd{postfix}_loss', v.item())
                        if log_with_wandb:
                            wandb.log({f'per_distr_shift/ood_{ood_n_shots}sh_{shift}_cnd{postfix}_loss': v.item()})

                # for ood_n_shots, sum_cnd in nshots2sum_cnd.items():
                    # mean_cnd = sum_cnd / len(self.shift2loaders)
                    # pl_module.log(f'ood_{ood_n_shots}sh_avg_cnd_loss', mean_cnd)
            for ood_n_shots, sum_cnd_dct in nshots2sum_cnd.items():
                for k, sum_cnd in sum_cnd_dct.items():
                    postfix = k.split('normed_dist_c_mean')[1]
                    mean_cnd = sum_cnd / len(self.shift2loaders)
                    pl_module.log(f'ood_{ood_n_shots}sh_avg_cnd{postfix}_loss', mean_cnd.item())
                    if log_with_wandb:
                        wandb.log({f'ood_{ood_n_shots}sh_avg_cnd{postfix}_loss': mean_cnd.item()})
            # remove temporary checkpoint
            os.remove(ckpt_path)

