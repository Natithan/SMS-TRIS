from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from experiments.datasets import Causal3DDataset

import pytorch_lightning as pl
import wandb
from models.shared.callbacks import BaselineCorrelationMetricsLogCallback
from models.shared.causal_encoder import get_causal_encoder
from models.shared.next_step_predictor import NextStepPredictor
from models.shared.rendered_view_predictor import RenderedViewPredictor
from models.shared.triplet_evaluator import TripletEvaluator
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
import os
from collections import OrderedDict, defaultdict

import sys

from util import CUDAfy
from math import prod
sys.path.append('../')
from models.shared import CosineWarmupScheduler, SineWarmupScheduler, get_act_fn, log_dict, Encoder, Decoder, SimpleEncoder, SimpleDecoder, TransitionPrior, TargetClassifier, CausalEncoder, ImageLogCallback, CorrelationMetricsLogCallback, SparsifyingGraphCallback
from models.shared import AutoregNormalizingFlow, gaussian_log_prob


# class CITRISVAE(pl.LightningModule):
class CITRISVAE(RenderedViewPredictor, TripletEvaluator):
    """ The main module implementing CITRIS-VAE """
    DEFAULT_VAL_TRACK_METRIC = 'val_comb_loss'
    def __init__(self, c_hid, num_latents, lr,
                 num_causal_vars,
                 warmup=100, max_iters=100000,
                 kld_warmup=0,
                 imperfect_interventions=False,
                 img_width=64,
                 c_in=3,
                 lambda_reg=0.01,
                 var_names=None,
                 causal_encoder_checkpoint=None,
                 classifier_num_layers=1,
                 classifier_act_fn='silu',
                 classifier_gumbel_temperature=1.0,
                 classifier_use_normalization=True,
                 classifier_use_conditional_targets=True,
                 classifier_momentum=0.9,
                 decoder_num_blocks=1,
                 act_fn='silu',
                 no_encoder_decoder=False,
                 use_flow_prior=True,
                 cluster_logging=False,
                 fullview_baseline=False, # Nathan
                 no_init_masking=False, # Nathan
                 no_context_masking=False, # Nathan
                 use_baseline_correlation_metrics=False, # Nathan
                 std_min=0,
                 fixed_logstd=False,
                 DataClass=None,
                 per_latent_full_silos=False,
                 sample_mean=False,
                 cheat_cfd=False,
                 standalone=True,
                 **kwargs):
        """
        Parameters
        ----------
        c_hid : int
                Hidden dimensionality to use in the network
        num_latents : int
                      Number of latent variables in the VAE
        lr : float
             Learning rate to use for training
        num_causal_vars : int
                          Number of causal variables / size of intervention target vector
        warmup : int
                 Number of learning rate warmup steps
        max_iters : int
                    Number of max. training iterations. Needed for 
                    cosine annealing of the learning rate.
        kld_warmup : int
                     Number of steps in the KLD warmup (default no warmup)
        imperfect_interventions : bool
                                  Whether interventions can be imperfect or not
        img_width : int
                    Width of the input image (assumed to be equal to height)
        c_in : int
               Number of input channels (3 for RGB)
        lambda_reg : float
                     Regularizer for promoting intervention-independent information to be modeled
                     in psi(0)
        var_names : list
                    List of names of the causal variables. Used for logging.
        causal_encoder_checkpoint : str
                                    Path to the checkpoint of a Causal-Encoder model to use for
                                    the triplet evaluation.
        classifier_num_layers : int
                                Number of layers to use in the target classifier network.
        classifier_act_fn : str
                            Activation function to use in the target classifier network.
        classifier_gumbel_temperature : float
                                        Temperature to use for the Gumbel Softmax sampling.
        classifier_use_normalization : bool
                                       Whether to use LayerNorm in the target classifier or not.
        classifier_use_conditional_targets : bool
                                             If True, we record conditional targets p(I^t+1_i|I^t+1_j)
                                             in the target classifier. Needed when intervention targets 
                                             are confounded.
        classifier_momentum_model : float
                                    Whether to use momentum or not in smoothing the target classifier
        decoder_num_blocks : int
                             Number of residual blocks to use per dimension in the decoder.
        act_fn : str
                 Activation function to use in the encoder and decoder network.
        no_encoder_decoder : bool
                             If True, no encoder or decoder are initialized. Used for CITRIS-NF
        """
        super().__init__(DataClass, num_causal_vars, num_latents)
        self.save_hyperparameters() # Automatically saves all arguments to self.hparams
        self.num_latents = self.hparams.num_latents
        act_fn_func = get_act_fn(self.hparams.act_fn)

        # region Encoder-Decoder init
        if self.hparams.no_encoder_decoder:
            self.encoder, self.decoder = nn.Identity(), nn.Identity()
        else:
            if self.hparams.img_width == 32:
                self.encoder = SimpleEncoder(num_input_channels=self.hparams.c_in,
                                             base_channel_size=self.hparams.c_hid,
                                             latent_dim=self.hparams.num_latents)
                self.decoder = SimpleDecoder(num_input_channels=self.hparams.c_in,
                                             base_channel_size=self.hparams.c_hid,
                                             latent_dim=self.hparams.num_latents)
            else:
                self.encoder = Encoder(num_latents=self.hparams.num_latents,
                                       c_hid=self.hparams.c_hid,
                                       c_in=self.hparams.c_in,
                                       width=self.hparams.img_width,
                                       act_fn=act_fn_func,
                                       variational=True)
                self.decoder = Decoder(num_latents=self.hparams.num_latents,
                                       c_hid=self.hparams.c_hid,
                                       c_out=self.hparams.c_in,
                                       width=self.hparams.img_width,
                                       num_blocks=self.hparams.decoder_num_blocks,
                                       act_fn=act_fn_func)
        # endregion
        # region Transition prior
        self.prior_t1 = TransitionPrior(num_latents=self.hparams.num_latents,
                                        num_blocks=self.hparams.num_causal_vars,
                                        c_hid=self.hparams.c_hid,
                                        imperfect_interventions=self.hparams.imperfect_interventions,
                                        lambda_reg=self.hparams.lambda_reg,
                                        autoregressive_model=self.hparams.autoregressive_prior,
                                        gumbel_temperature=self.hparams.classifier_gumbel_temperature,
                                        fullview_baseline=self.hparams.fullview_baseline,
                                        no_init_masking=self.hparams.no_init_masking,
                                        no_context_masking=self.hparams.no_init_masking,
                                        std_min=self.hparams.std_min,
                                        fixed_logstd=self.hparams.fixed_logstd,
                                        per_latent_full_silos=self.hparams.per_latent_full_silos)
        # endregion
        # region Target classifier
        self.intv_classifier = TargetClassifier(num_latents=self.hparams.num_latents,
                                                num_blocks=self.hparams.num_causal_vars,
                                                c_hid=self.hparams.c_hid,
                                                num_layers=self.hparams.classifier_num_layers,
                                                act_fn=get_act_fn(self.hparams.classifier_act_fn),
                                                var_names=self.hparams.var_names,
                                                momentum_model=self.hparams.classifier_momentum,
                                                gumbel_temperature=self.hparams.classifier_gumbel_temperature,
                                                use_normalization=self.hparams.classifier_use_normalization,
                                                use_conditional_targets=self.hparams.classifier_use_conditional_targets,
                                                fullview_baseline=self.hparams.fullview_baseline)
        # endregion
        #region Flow
        if self.hparams.use_flow_prior:
            self.flow = AutoregNormalizingFlow(self.hparams.num_latents,
                                               num_flows=4,
                                               act_fn=nn.SiLU,
                                               hidden_per_var=16)
        #endregion
        # Warmup scheduler for KL (if selected)
        self.kld_scheduler = SineWarmupScheduler(kld_warmup)
        # region Load causal encoder for triplet evaluation
        self.causal_encoder, self.causal_encoder_true_epoch = get_causal_encoder(self.hparams.causal_encoder_checkpoint)
        # if self.hparams.causal_encoder_checkpoint is not None:
        #     self.causal_encoder_true_epoch = int(1e5)  # We want to log the true causal encoder distance once
        #     self.causal_encoder = CausalEncoder.load_from_checkpoint(self.hparams.causal_encoder_checkpoint)
        #     for p in self.causal_encoder.parameters():
        #         p.requires_grad_(False)
        # else:
        #     self.causal_encoder = None
        # endregion
        # Logging
        self.output_to_input = None

        # if 'DataClass' in kwargs:
        #     DataClass = kwargs['DataClass']
        # else:
        #     raise ValueError('DataClass not provided to model init')
        self.DA_loss_factor_keys = ['x_mean', 'y_sum', 'z_sum', 'zpsi_mean', 'c_mean'] + \
                                   [f'zpsi_{f}' for f in DataClass.SHORT_INTERVENED_FACTORS_AND_TRASH] + \
                                   [f'c_{f}' for f in DataClass.SHORT_FACTORS]


        if cheat_cfd:
            raise NotImplementedError('DEPRECATED: doing this in separate training stage')
            self.z_to_c_predictor = CausalEncoder(c_hid=128,
                                                  lr=4e-3,
                                                  causal_var_info=OrderedDict(
                                                      {k: v for (k, v) in DataClass.MY_VAR_INFO.items() if
                                                       k in DataClass.FACTORS}),
                                                  single_linear=True,
                                                  c_in=self.num_fine_vars*2,
                                                  warmup=0)

    def forward(self, x):
        # Full encoding and decoding of samples
        z_mean, z_logstd = self.encoder(x)
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample)
        return x_rec, z_sample, z_mean, z_logstd

    def encode(self, x, random=True):
        # Map input to encoding, e.g. for correlation metrics
        z_mean, z_logstd = self.encoder(x)
        if random:
            z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        else:
            z_sample = z_mean
        # Paper: Similarly to the encoder distribution, one can make the prior more flexible by using normalizing flows.
        # However, in experiments, this showed to not provide any improvement compared to increased computational cost and parameters
        if self.hparams.use_flow_prior:
            z_sample, _ = self.flow(z_sample)
        return z_sample

    def configure_optimizers(self, max_iters=None):
        # We use different learning rates for the target classifier (higher lr for faster learning).
        intv_params, other_params = [], []
        for name, param in self.named_parameters():
            if name.startswith('intv_classifier'):
                intv_params.append(param)
            else:
                other_params.append(param)
        optimizer = optim.AdamW([{'params': intv_params, 'lr': self.hparams.classifier_lr, 'weight_decay': 1e-4},
                                 {'params': other_params}], lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters if max_iters is None else max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, batch, mode='train'):
        """ Main training method for calculating the loss """
        imgs, labels, target = self._get_loss_batch_unpack(batch)
        # En- and decode every element of the sequence, except first element no decoding
        z_mean, z_logstd = self.encoder(imgs.flatten(0, 1))
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample.unflatten(0, imgs.shape[:2])[:, 1:].flatten(0, 1)) # x_{t+1}
        z_sample, z_mean, z_logstd, x_rec = [t.unflatten(0, (imgs.shape[0], -1)) for t in
                                             [z_sample, z_mean, z_logstd, x_rec]]

        if self.hparams.use_flow_prior:
            init_nll = -gaussian_log_prob(z_mean[:, 1:], z_logstd[:, 1:], z_sample[:, 1:]).sum(dim=-1) # log probs assigned by the encoder to the latents it sampled itself
            z_sample, ldj = self.flow(z_sample.flatten(0, 1))
            z_sample = z_sample.unflatten(0, (imgs.shape[0], -1))
            ldj = ldj.unflatten(0, (imgs.shape[0], -1))[:, 1:]
            out_nll = self.prior_t1.sample_based_nll(z_t1=z_sample[:, None, 1:].flatten(0, 1), # Will calculate prob for: latent of second image {t+1 aka t1}
                                                     target=target.flatten(0, 1),
                                                     z_t=z_sample[:, None, :-1].flatten(0, 1)) # Based on intervention targets (target) and latents of first image {t}
            out_nll = out_nll.unflatten(0, (imgs.shape[0], -1))
            p_z = out_nll
            p_z_x = init_nll - ldj
            kld = -(p_z_x - p_z)
            kld_t1_all = kld.unflatten(0, (imgs.shape[0], -1)).sum(dim=1)
        else:
            # Calculate KL divergence between every pair of frames
            kld_t1_all = self.prior_t1.kl_divergence(z_t=z_mean[:, :-1].flatten(0, 1),
                                                     target=target.flatten(0, 1),
                                                     z_t1_mean=z_mean[:, 1:].flatten(0, 1),
                                                     z_t1_logstd=z_logstd[:, 1:].flatten(0, 1),
                                                     z_t1_sample=z_sample[:, 1:].flatten(0, 1))
            kld_t1_all = kld_t1_all.unflatten(0, (imgs.shape[0], -1)).sum(dim=1)

        # Calculate reconstruction loss
        if isinstance(self.decoder, nn.Identity):
            rec_loss = z_mean.new_zeros(imgs.shape[0], imgs.shape[1])
        else:
            rec_loss = F.mse_loss(x_rec, labels[:, 1:], reduction='none').sum(dim=[-3, -2, -1])
        # Combine to full loss
        kld_factor = self.kld_scheduler.get_factor(self.global_step)
        loss = (kld_factor * (kld_t1_all * self.hparams.beta_t1) + rec_loss.sum(dim=1)).mean()
        loss = loss / (imgs.shape[1] - 1)
        # Add target classifier loss
        loss_model, loss_z = self.intv_classifier(z_sample=z_sample,
                                                  logger=self if not self.hparams.cluster_logging else None,
                                                  target=target,
                                                  transition_prior=self.prior_t1)
        loss = loss + (loss_model + loss_z) * self.hparams.beta_classifier

        # Logging
        self.log(f'{mode}_kld_t1', kld_t1_all.mean() / (imgs.shape[1] - 1))
        self.log(f'{mode}_rec_loss_t1', rec_loss.mean())
        if mode == 'train':
            self.log(f'{mode}_kld_scheduling', kld_factor)
        self.log(f'{mode}_intv_classifier_model', loss_model)
        self.log(f'{mode}_intv_classifier_z', loss_z)

        return loss


    def triplet_prediction(self, imgs, source, use_cheating_target_assignment=False):
        """ Generates the triplet prediction of input image pairs and causal mask """
        input_imgs = imgs[:, :2].flatten(0,
                                         1)  # To plot, use plot_tensor_as_img((imgs[0].permute(1,2,0,3).reshape(3,64,3*64).permute(1,2,0) + 1)/2)
        z_mean, z_logstd = self.encoder(input_imgs)
        input_samples = z_mean
        input_samples = input_samples.unflatten(0, (-1, 2))
        # Map the causal mask to a latent variable mask
        target_assignment = self.get_maybe_cheating_target_assignment(use_cheating_target_assignment)
        if source.shape[-1] + 1 == target_assignment.shape[-1]:  # No-variables missing
            source = torch.cat([source, source[..., -1:] * 0.0], dim=-1)
        elif target_assignment.shape[-1] > source.shape[-1]:
            target_assignment = target_assignment[..., :source.shape[-1]]
        # Take the latent variables from image 1 respective to the mask, and image 2 the inverse
        mask_1 = (target_assignment[None, :, :] * (1 - source[:, None, :])).sum(dim=-1)
        mask_2 = 1 - mask_1
        triplet_samples = mask_1 * input_samples[:, 0] + mask_2 * input_samples[:, 1]
        # Decode the new combination
        triplet_rec = self.decoder(triplet_samples)
        if self.output_to_input is not None:
            triplet_rec = self.output_to_input(triplet_rec)
        return triplet_rec

    def get_maybe_cheating_target_assignment(self, use_cheating_target_assignment):
        target_assignment = self.prior_t1.get_target_assignment(
            hard=True) if (not use_cheating_target_assignment) else self.last_target_assignment.to(self.prior_t1.get_target_assignment(
            hard=True).device)
        return target_assignment

    # MOVED TO TRIPLET_EVALUATOR.PY
    # def triplet_evaluation(self, batch, mode='val',dataloader_idx=None, use_cheating_target_assignment=False):
    #     """ Evaluates the triplet prediction for a batch of images """
    #     # Input handling
    #     # if len(batch) == 2:
    #     #     imgs, source = batch
    #     #     labels = imgs
    #     #     latents = None
    #     #     obj_indices = None
    #     # elif len(batch) == 3 and len(batch[1].shape) == 2:
    #     #     imgs, source, latents = batch
    #     #     labels = imgs
    #     #     obj_indices = None
    #     # elif len(batch) == 3:
    #     #     imgs, labels, source = batch
    #     #     obj_indices = None
    #     #     latents = None
    #     # elif len(batch) == 4 and len(batch[-1].shape) > 1:
    #     #     imgs, labels, source, latents = batch
    #     #     obj_indices = None
    #     # elif len(batch) == 4 and len(batch[-1].shape) == 1:
    #     #     imgs, source, latents, obj_indices = batch
    #     #     labels = imgs
    #     # Input handling for refactored code that returns a dict instead of a list
    #     imgs = batch['encs']
    #     source = batch['targets']
    #     latents = batch['lat'] if 'lat' in batch else None
    #     obj_indices = batch['obj_triplet_indices'] if 'obj_triplet_indices' in batch else None
    #     labels = batch['labels'] if 'labels' in batch else imgs
    #
    #     triplet_label = labels[:, -1]
    #     # Estimate triplet prediction
    #     triplet_rec = self.triplet_prediction(imgs, source, use_cheating_target_assignment)
    #
    #     if self.causal_encoder is not None and latents is not None:
    #         self.causal_encoder.eval()
    #         # Evaluate the causal variables of the predicted output
    #         with torch.no_grad():
    #             losses, dists, norm_dists, v_dict = self.causal_encoder.get_distances(triplet_rec, latents[:, -1],
    #                                                                                   return_norm_dists=True,
    #                                                                                   return_v_dict=True)
    #             self.all_v_dicts.append(v_dict)
    #             rec_loss = sum([norm_dists[key].mean() for key in losses])
    #             mean_loss = sum([losses[key].mean() for key in losses])
    #             postfix = '_cheating' if use_cheating_target_assignment else ''
    #             self.mylog(f'{mode}_distance_loss{postfix}', mean_loss, dataloader_idx=dataloader_idx)
    #             for key in dists:
    #                 self.all_val_dists[key].append(dists[key])
    #                 self.mylog(f'{mode}_{key}_dist{postfix}', dists[key].mean(), dataloader_idx=dataloader_idx)
    #                 self.mylog(f'{mode}_{key}_norm_dist{postfix}', norm_dists[key].mean(), dataloader_idx=dataloader_idx)
    #                 if obj_indices is not None:  # For excluded object shapes, record results separately
    #                     for v in obj_indices.unique().detach().cpu().numpy():
    #                         self.mylog(f'{mode}_{key}_dist_obj_{v}{postfix}', dists[key][obj_indices == v].mean(), dataloader_idx=dataloader_idx)
    #                         self.mylog(f'{mode}_{key}_norm_dist_obj_{v}{postfix}', norm_dists[key][obj_indices == v].mean(), dataloader_idx=dataloader_idx)
    #             if obj_indices is not None:
    #                 self.all_val_dists['object_indices'].append(obj_indices)
    #             if self.current_epoch > 0 and self.causal_encoder_true_epoch >= self.current_epoch:
    #                 self.causal_encoder_true_epoch = self.current_epoch
    #                 if len(triplet_label.shape) == 2 and hasattr(self, 'autoencoder'):
    #                     triplet_label = self.autoencoder.decoder(triplet_label)
    #                 _, true_dists = self.causal_encoder.get_distances(triplet_label, latents[:, -1])
    #                 for key in dists:
    #                     self.mylog(f'{mode}_{key}_true_dist{postfix}', true_dists[key].mean(), dataloader_idx=dataloader_idx)
    #     else:
    #         rec_loss = torch.zeros(1, )
    #
    #     return rec_loss

    # def get_y_mseloss(self, batch):
    #     raise NotImplementedError("Not implemented for CITRISVAE")



    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='train')
        self.log('train_loss', loss)
        return loss



    # def validation_step(self, batch, batch_idx, dataloader_idx=0):
    #     if dataloader_idx == 0:
    #         imgs, *_ = batch
    #         loss = self.triplet_evaluation(batch, mode='val', dataloader_idx=dataloader_idx)
    #         self.mylog('val_loss', loss, dataloader_idx)
    #     else:
    #         self.log_mse_losses(batch, dataloader_idx, split='val')
    #
    # def test_step(self, batch, batch_idx, dataloader_idx=0):
    #     if dataloader_idx == 0:
    #         imgs, *_ = batch # batch: imgs [B, 3, C, H,W], targets [B,num_targets], true_latents [B,3, total_target_dims]
    #         loss = self.triplet_evaluation(batch, mode='test')
    #         self.log('test_loss', loss)
    #     else:
    #         self.log_mse_losses(batch, dataloader_idx, split='test')




    @classmethod
    def get_specific_callbacks(cls, exmp_triplet_inputs, exmp_fp_inputs, dataset, cluster, **kwargs):
        callbacks = []
        callbacks.append(ImageLogCallback(exmp_triplet_inputs, dataset, every_n_epochs=kwargs['img_callback_freq'], cluster=cluster, exmp_fp_inputs=exmp_fp_inputs)) # Nathan: not sure for what purpose images were not being logged by NF, so changed it.
        return callbacks


    # DEPRECATED: refactored into superclass
    # @staticmethod
    # def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, correlation_dataset=None, correlation_test_dataset=None, exmp_fp_inputs=None, mini=False, use_baseline_correlation_metrics=False, **kwargs):
    #     callbacks = []
    #     # Create learning rate callback
    #     callbacks.append(LearningRateMonitor('step'))
    #     if kwargs['add_enco_sparsification']:
    #         callbacks.append(SparsifyingGraphCallback(dataset=dataset, cluster=cluster,mini=mini)) # Nathan: added this
    #     if not kwargs['no_alignment_measuring']:
    #         if use_baseline_correlation_metrics:
    #             corrmet_callback_class = BaselineCorrelationMetricsLogCallback
    #             ignore_prior_t1 = True
    #         else:
    #             corrmet_callback_class = CorrelationMetricsLogCallback
    #             ignore_prior_t1 = False
    #         # if mini:
    #         #     callbacks.append(corrmet_callback_class(correlation_dataset, cluster=cluster, test_dataset=correlation_test_dataset, num_train_epochs=5, ignore_prior_t1=ignore_prior_t1))
    #         # else:
    #         #     callbacks.append(corrmet_callback_class(correlation_dataset, cluster=cluster, test_dataset=correlation_test_dataset, ignore_prior_t1=ignore_prior_t1))
    #         cormet_args = {
    #             "dataset": correlation_dataset,
    #             "cluster": cluster,
    #             "test_dataset": correlation_test_dataset,
    #         }
    #     # img_callback = ImageLogCallback(exmp_inputs, dataset, every_n_epochs=10 if not cluster else 50, cluster=cluster)
    #     callbacks.append(ImageLogCallback(exmp_inputs, dataset, every_n_epochs=kwargs['img_callback_freq'], cluster=cluster, exmp_fp_inputs=exmp_fp_inputs)) # Nathan: not sure for what purpose images were not being logged by NF, so changed it.
    #     return callbacks

    def sample_zt1(self, z_t, I_t1, use_mean_to_autoregress=None, return_mean_as_sample=True, also_return_params=False, overriden_fixed_logstd=None):
        if use_mean_to_autoregress is None:
            use_mean_to_autoregress = self.hparams.sample_mean
        psi = self.prior_t1.get_target_assignment(hard=True)
        psi_intv = psi[:, :-1]
        if self.hparams.autoregressive_prior:
            input_ready_I_t1 = torch.cat([I_t1.flatten(0, 1), torch.zeros_like(I_t1[:, 0, 0:1])], dim=1)

            inp = torch.zeros_like(z_t)
            for i in range(self.hparams.num_latents):
                zt1_mean, zt1_logstd = self.prior_t1._get_prior_params(
                    z_t,
                    target=input_ready_I_t1,
                    target_samples=psi.expand(z_t.shape[0], -1, -1),
                    z_t1=inp,
                    overriden_fixed_logstd=overriden_fixed_logstd)
                zt1_mean, zt1_logstd = [(zt1_x * psi.transpose(0,1)[None]).sum(dim=1) for zt1_x in [zt1_mean, zt1_logstd]]
                # zt1_s = zt1_mean_collapsed + zt1_logstd_collapsed.exp() * torch.randn_like(zt1_logstd_collapsed) # TODO given that .init_layer takes 2*num_latents as input, maybe I should directly give mean and std instead of a sample
                zt1_s = zt1_mean + 0.0 # 0.0 to make sure it's a new tensor
                if not use_mean_to_autoregress:
                    zt1_s += zt1_logstd.exp() * torch.randn_like(zt1_logstd) # Other places where sampling happens: citrisnf/lightning_module.py/get_losses
                z_t1_slice = torch.zeros_like(zt1_s)
                z_t1_slice[:, i] = zt1_s[:, i]
                inp = inp + z_t1_slice
        else:
            if not self.hparams.fullview_baseline:
                zt1_intv_mean, zt1_intv_logstd = self.prior_t1._get_intv_params(z_t.shape, target=None)
                zt1_intv_s = zt1_intv_mean + zt1_intv_logstd * torch.randn_like(zt1_intv_logstd)
                zt1_intv_s_m = zt1_intv_s * I_t1 * psi_intv
                zt1_intv_sms = zt1_intv_s_m.sum(dim=-1)

                zt1_nat_mean, zt1_nat_logstd = self.prior_t1._get_prior_params(z_t)
                zt1_nat_s = zt1_nat_mean + zt1_nat_logstd * torch.randn_like(zt1_nat_logstd)

                intv_z_mask = (I_t1 * psi_intv).sum(dim=-1).to(torch.bool)
                combined_zt1 = torch.where(intv_z_mask, zt1_intv_sms, zt1_nat_s)
                return combined_zt1
            else:
                input_ready_I_t1 = torch.cat([I_t1.flatten(0,1),torch.zeros_like(I_t1[:,0,0:1])],dim=1)
                zt1_mean, zt1_logstd = self.prior_t1._get_prior_params(z_t,target=input_ready_I_t1)
                zt1_s = zt1_mean + zt1_logstd * torch.randn_like(zt1_logstd)
        sample = zt1_s if not return_mean_as_sample else zt1_mean
        if also_return_params:
            return sample, zt1_mean, zt1_logstd
        else:
            return sample

    def sample_yt1(self, *args): # For VAE, Y and Z are the same
        return self.sample_zt1(*args)


    def sample_xt1(self, x_t, I_t1, also_z_mse=False, true_x_t1=None):
        z_t = self.encode(x_t,random=False)
        z_t1 = self.sample_zt1(z_t, I_t1, use_mean_to_autoregress=True)
        if also_z_mse:
            true_z_t1 = self.encode(true_x_t1,random=False)
            z_mse = F.mse_loss(z_t1, true_z_t1)
        x_t1 = self.decode(z_t1)
        if also_z_mse:
            return x_t1, z_mse
        return x_t1


    def get_grouped_flp_params(self, use_maxcorr_psi=None):
        # # params = {'shared': []} | {i: [] for i in range(self.num_fine_vars)} # no shared params atm
        # params = {i: [] for i in range(self.num_fine_vars)}
        # for name, param in self.named_parameters():
        #     if any([substr in name for substr in ['encoder','e_to_c']]):
        #         continue
        #     elif any([substr in name for substr in ['context_layer', 'target_layer', 'net']]):
        #         for i, param_slice in enumerate(param):
        #             params[i].append(param_slice.clone())
        #     else:
        #         raise ValueError(f'Unknown parameter group for {name}')
        # return params


        # psi_idx = self.hparams.DataClass.SHORT_COARSE_FACTORS_AND_TRASH.index(unfreeze_factor.split('uf_')[1]) if unfreeze_factor != 'uf_all' else 'uf_all' # number between 0 and num_factors
        # psi = self.prior_t1.get_target_assignment(hard=True) # shape (num_latents, num_factors)
        # if unfreeze_factor != 'uf_all':
        #     unfreeze_idxs = torch.nonzero(psi[:, psi_idx]).flatten()
        #
        # mask_list = []
        # for name, param in self.named_parameters():
        #     if unfreeze_factor == 'uf_all':
        #         param_mask = torch.ones_like(param)
        #     else:
        #         param_mask = torch.zeros_like(param)
        #         if 'net.3' in name: # The last layer, and the only one that contains completely specialized parameters
        #             per_factor_dim = param.shape[0] // self.hparams.num_latents
        #             for idx in unfreeze_idxs:
        #                 param_mask[idx * per_factor_dim:(idx + 1) * per_factor_dim] = 1
        #     mask_list.append(param_mask)
        # return mask_list
        # psi = self.psi
        psi, uses_maxcorr = self.get_learnt_or_maxcorr_psi(use_maxcorr=use_maxcorr_psi, also_return_updated_use_maxcorr=True)
        if not uses_maxcorr:
            groups = self.hparams.DataClass.SHORT_INTERVENED_FACTORS_AND_TRASH
        else:
            groups = self.hparams.DataClass.SHORT_COARSE_FACTORS
        params = {'shared': []} | {f: [] for f in groups}
        for name, param in self.prior_t1.named_parameters():
            if 'net.3' in name:
                slice_idx_total = 0
                for f in groups:
                    slice_idxs = torch.nonzero(psi[:, groups.index(f)]).flatten()
                    slice_idx_total += len(slice_idxs)
                    per_factor_dim = param.shape[0] // self.hparams.num_latents
                    if len(slice_idxs) > 0:
                        param_slice = torch.cat([param[idx * per_factor_dim:(idx + 1) * per_factor_dim] for idx in slice_idxs])
                        params[f].append(param_slice.clone())
                assert slice_idx_total == psi.shape[0]
            else:
                params['shared'].append(param.clone())
        return params

    def get_sparse_update_params(self):
        return ((n, p) for n, p in self.prior_t1.named_parameters() if not 'target_params' in n) # target_params is psi, which is not updated


    def is_shared(self, param_name):
        if not self.hparams.per_latent_full_silos:
            return 'net.3' not in param_name # The last layer, and the only one that contains completely specialized parameters
        else:
            return False

    @property
    def psi(self):
        return self.prior_t1.get_target_assignment(hard=True).float()

    def get_psi_params(self):
        return self.prior_t1.target_params


    @property
    def prior_parts(self):
        return [self.prior_t1]

    def get_target_assignment(self, hard=False):
        return self.prior_t1.get_target_assignment(hard=hard)

    def maybe_add_intv_loss(self, loss, loss_dict, ztt1_sample, It1, args=None, grad_for_idloss=False):
        # with torch.no_grad():
        with (nullcontext() if grad_for_idloss else torch.no_grad()):
            loss_model, loss_z = self.intv_classifier(z_sample=ztt1_sample,
                                                      logger=self if not self.hparams.cluster_logging else None,
                                                      target=It1,
                                                      transition_prior=self.prior_t1)
        loss_dict['intv_classifier_model'] = loss_model
        loss_dict['ID_loss'] = loss_dict['ID_loss'] + (loss_model + loss_z) * self.hparams.beta_classifier

    def get_kld(self, It1, ztt1_logstd, ztt1_mean, ztt1_sample,overriden_fixed_logstd=None):
        # Calculate KL divergence between every pair of frames
        kld_t1_all, (prior_mean_per_hyp, prior_logstd_per_hyp) = self.prior_t1.kl_divergence(
            z_t=ztt1_mean[:, :-1].flatten(0, 1),
            target=It1.flatten(0, 1),
            z_t1_mean=ztt1_mean[:, 1:].flatten(0, 1),
            z_t1_logstd=ztt1_logstd[:, 1:].flatten(0, 1),
            z_t1_sample=ztt1_sample[:, 1:].flatten(0, 1),
            also_prior_params=True,
        overriden_fixed_logstd=overriden_fixed_logstd)
        kld_t1_all = kld_t1_all.unflatten(0, (It1.shape[0], -1)).sum(dim=1)
        return kld_t1_all