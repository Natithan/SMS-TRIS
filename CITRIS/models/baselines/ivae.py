"""
Model architectures for iVAE* and SlowVAE
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from models.shared.rendered_view_predictor import RenderedViewPredictor
from models.shared.triplet_evaluator import TripletEvaluator
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
from collections import defaultdict

import sys
sys.path.append('../')
from models.shared import CosineWarmupScheduler, SineWarmupScheduler, get_act_fn, kl_divergence, gaussian_log_prob, log_dict, Encoder, Decoder, SimpleEncoder, SimpleDecoder
from models.shared import ImageLogCallback, CausalEncoder, AutoregLinear
from models.ae import Autoencoder
from models.shared.callbacks import BaselineCorrelationMetricsLogCallback
from models.shared import AutoregNormalizingFlow
from util import tn


class iVAE(RenderedViewPredictor, TripletEvaluator):
    """
    Module for implementing the adapted iVAE 
    It is similarly structured as CITRIS-VAE, although being reduced
    to a standard VAE instead of with a full transition prior etc.
    """
    DEFAULT_VAL_TRACK_METRIC = 'val_comb_loss_grouped_latents'
    def __init__(self, c_hid, num_latents, lr, 
                       num_causal_vars,
                       c_in=3,
                       warmup=100, 
                       max_iters=100000,
                       kld_warmup=100,
                       beta_t1=1.0,
                       img_width=32,
                       decoder_num_blocks=1,
                       act_fn='silu',
                       causal_var_info=None,
                       causal_encoder_checkpoint=None,
                       use_flow_prior=True,
                       autoencoder_checkpoint=None,
                       autoregressive_prior=False,
                 DataClass=None,
                 fixed_logstd=False,
                 sample_mean=False,
                       **kwargs):
        super().__init__(DataClass, num_causal_vars, num_latents)
        self.save_hyperparameters()
        
        act_fn_func = get_act_fn(self.hparams.act_fn)
        if autoencoder_checkpoint is None:
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
                                          use_batch_norm=False)
                self.decoder = Decoder(num_latents=self.hparams.num_latents,
                                          c_hid=self.hparams.c_hid,
                                          c_out=self.hparams.c_in,
                                          width=self.hparams.img_width,
                                          num_blocks=self.hparams.decoder_num_blocks,
                                          act_fn=act_fn_func,
                                          use_batch_norm=False)
        else:
            self.autoencoder = Autoencoder.load_from_checkpoint(self.hparams.autoencoder_checkpoint)
            for p in self.autoencoder.parameters():
                p.requires_grad_(False)
            def encod_function(inp):
                mean = self.autoencoder.encoder(inp)
                log_std = torch.zeros_like(mean).fill_(np.log(self.autoencoder.hparams.noise_level))
                return mean, log_std
            self.encoder = encod_function
            self.decoder = lambda inp: self.autoencoder.decoder(inp)
        # Prior of p(z^t+1|z^t,I^t)
        if self.hparams.autoregressive_prior:
            self.prior_net_cond = nn.Linear(num_latents + num_causal_vars, 16 * num_latents)
            self.prior_net_init = AutoregLinear(num_latents, 1, 16, diagonal=False)
            self.prior_net_head = nn.Sequential(
                    nn.SiLU(),
                    AutoregLinear(num_latents, 16, 16, diagonal=True),
                    nn.SiLU(),
                    AutoregLinear(num_latents, 16, 2, diagonal=True)
                )
        else:
            self.prior_net = nn.Sequential(
                    nn.Linear(num_latents + num_causal_vars, self.hparams.c_hid*8),
                    nn.SiLU(),
                    nn.Linear(self.hparams.c_hid*8, self.hparams.c_hid*8),
                    nn.SiLU(),
                    nn.Linear(self.hparams.c_hid*8, num_latents*2)
                )
        if self.hparams.use_flow_prior:
            self.flow = AutoregNormalizingFlow(self.hparams.num_latents,
                                               num_flows=4,
                                               act_fn=nn.SiLU,
                                               hidden_per_var=16)
        self.kld_scheduler = SineWarmupScheduler(kld_warmup, start_factor=0.01)
        # Causal Encoder loading
        if self.hparams.causal_encoder_checkpoint is not None:
            self.causal_encoder_true_epoch = int(1e5)  # We want to log the true causal encoder distance once
            self.causal_encoder = CausalEncoder.load_from_checkpoint(self.hparams.causal_encoder_checkpoint)
            for p in self.causal_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.causal_encoder = None
        self.target_assignment = None
        self.output_to_input = None
        self.all_val_dists = defaultdict(list)

    def forward(self, x):
        z_mean, z_logstd = self.encoder(x)
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample)
        return x_rec, z_sample, z_mean, z_logstd

    def encode(self, x, random=True):
        z_mean, z_logstd = self.encoder(x)
        if random:
            z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        else:
            z_sample = z_mean
        if self.hparams.use_flow_prior:
            z_sample, _ = self.flow(z_sample)
        return z_sample

    def _run_prior(self, ztt1_sample, It1):
        inp = torch.cat([ztt1_sample[:, :-1], It1], dim=-1)
        true_out = ztt1_sample[:, 1:]
        inp = inp.flatten(0, 1)
        true_out = true_out.flatten(0, 1)
        if self.hparams.autoregressive_prior:
            cond_feats = self.prior_net_cond(inp)
            init_feats = self.prior_net_init(true_out)
            comb_feats = cond_feats + init_feats
            out_feats = self.prior_net_head(comb_feats)
            out_feats = out_feats.unflatten(-1, (-1, 2))
            prior_mean, prior_logstd = out_feats.unbind(dim=-1)
        else:
            prior_mean, prior_logstd = self.prior_net(inp).chunk(2, dim=-1)
        prior_mean = prior_mean.unflatten(0, It1.shape[:2])
        prior_logstd = prior_logstd.unflatten(0, It1.shape[:2])
        if self.hparams.fixed_logstd: # To be in line with CITRISVAE/CITRISNF behaviour
            prior_logstd = torch.zeros_like(prior_logstd)
        return prior_mean, prior_logstd

    def sample_latent(self, batch_size, for_intervention=False):
        sample = torch.randn(batch_size, self.hparams.num_latents, device=self.device)
        return sample

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, batch, mode='train'):
        imgs, labels, target = self._get_loss_batch_unpack(batch)
        z_mean, z_logstd = self.encoder(imgs.flatten(0, 1))
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample.unflatten(0, imgs.shape[:2])[:,1:].flatten(0, 1))
        z_sample, z_mean, z_logstd, x_rec = [t.unflatten(0, (imgs.shape[0], -1)) for t in [z_sample, z_mean, z_logstd, x_rec]]

        if not self.hparams.use_flow_prior:
            # prior_t1_mean, prior_t1_logstd = self.prior_net(torch.cat([z_sample[:,:-1], target], dim=-1)).chunk(2, dim=-1)
            kld = self.get_kld(target, z_logstd, z_mean, z_sample)
        else:
            init_nll = -gaussian_log_prob(z_mean, z_logstd, z_sample)[:,1:].sum(dim=-1)
            z_sample, ldj = self.flow(z_sample.flatten(0, 1))
            z_sample = z_sample.unflatten(0, (imgs.shape[0], -1))
            ldj = ldj.unflatten(0, (imgs.shape[0], -1))[:,1:]
            # prior_t1_mean, prior_t1_logstd = self.prior_net(torch.cat([z_sample[:,:-1], target], dim=-1)).chunk(2, dim=-1)
            prior_t1_mean, prior_t1_logstd = self._run_prior(z_sample, target)
            out_nll = -gaussian_log_prob(prior_t1_mean, prior_t1_logstd, z_sample[:,1:]).sum(dim=-1)
            p_z = out_nll 
            p_z_x = init_nll - ldj
            kld = -(p_z_x - p_z).sum(dim=1)

        rec_loss = F.mse_loss(x_rec, labels[:,1:], reduction='none').sum(dim=list(range(2, len(x_rec.shape))))
        kld_factor = self.kld_scheduler.get_factor(self.global_step)
        loss = (kld_factor * kld * self.hparams.beta_t1 + rec_loss.sum(dim=1)).mean()
        loss = loss / (imgs.shape[1] - 1)

        self.log(f'{mode}_kld', kld.mean() / (imgs.shape[1]-1))
        self.log(f'{mode}_rec_loss', rec_loss.mean())
        if mode == 'train':
            self.log(f'{mode}_kld_scheduling', kld_factor)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='train')
        self.log('train_loss', loss)
        return loss

    def eval_step(self, batch, batch_idx, dataloader_idx=0, stage='val'):
        self.eval()
        super().eval_step(batch, batch_idx, dataloader_idx=dataloader_idx,stage=stage, only_cheating=True)



    def triplet_prediction(self, imgs, source, use_cheating_target_assignment):
        input_imgs = imgs[:,:2].flatten(0, 1)
        z_mean, z_logstd = self.encoder(input_imgs)
        if self.hparams.use_flow_prior:
            z_mean, _ = self.flow(z_mean)
        input_samples = z_mean
        input_samples = input_samples.unflatten(0, (-1, 2))
        
        target_assignment = self.target_assignment.to(z_mean.device)
        if target_assignment.shape[-1] > source.shape[-1]:
            target_assignment = target_assignment[...,:source.shape[-1]]
        mask_1 = (target_assignment[None,:,:] * (1 - source[:,None,:])).sum(dim=-1)
        mask_2 = 1 - mask_1
        triplet_samples = mask_1 * input_samples[:,0] + mask_2 * input_samples[:,1]
        if self.hparams.use_flow_prior:
            triplet_samples = self.flow.reverse(triplet_samples)
        triplet_rec = self.decoder(triplet_samples)
        if self.output_to_input is not None:
            triplet_rec = self.output_to_input(triplet_rec)
        return triplet_rec

    @classmethod
    def get_specific_callbacks(cls, exmp_triplet_inputs, exmp_fp_inputs, dataset, cluster, **kwargs):
        callbacks = []
        callbacks.append(ImageLogCallback(exmp_triplet_inputs, dataset, every_n_epochs=kwargs['img_callback_freq'], cluster=cluster, exmp_fp_inputs=exmp_fp_inputs)) # Nathan: not sure for what purpose images were not being logged by NF, so changed it.
        return callbacks

    @classmethod
    def add_cormet_callbacks(cls, cluster, correlation_dataset, correlation_test_dataset, mini,
                             use_baseline_correlation_metrics
                             ):
        cormet_args = {
            "dataset": correlation_dataset,
            "cluster": cluster,
            "test_dataset": correlation_test_dataset,
            "ignore_learnt_psi": True,
        }
        if mini:
            cormet_args['num_train_epochs'] = 5
        cormet_args |= cls.get_specific_cormet_args()
        return [BaselineCorrelationMetricsLogCallback(**cormet_args)]


    def maybe_add_intv_loss(self, *args, **kwargs):
        pass

    def get_kld(self, It1, ztt1_logstd, ztt1_mean, ztt1_sample, overriden_fixed_logstd=None):
        prior_t1_mean, prior_t1_logstd = self._run_prior(ztt1_sample, It1)
        kld = kl_divergence(ztt1_mean[:, 1:], ztt1_logstd[:, 1:], prior_t1_mean, prior_t1_logstd).sum(dim=[1, -1])
        return kld



    def sample_zt1(self, z_t, I_t1, use_mean_to_autoregress=None, return_mean_as_sample=True, also_return_params=False):
        if use_mean_to_autoregress is None:
            use_mean_to_autoregress = self.hparams.sample_mean
        inp = torch.cat([z_t, I_t1.flatten(0, 1)], dim=-1)
        if self.hparams.autoregressive_prior:

            cond_feats = self.prior_net_cond(inp)
            pred_zt1 = torch.zeros_like(z_t)
            for i in range(self.hparams.num_latents):
                init_feats = self.prior_net_init(pred_zt1)
                comb_feats = cond_feats + init_feats
                out_feats = self.prior_net_head(comb_feats)
                out_feats = out_feats.unflatten(-1, (-1, 2))
                zt1_mean, zt1_logstd = out_feats.unbind(dim=-1)
                clipped_zt1_logstd = zt1_logstd.clamp(-20, 20) # Nathan: clipping to avoid nan
                zt1_s = zt1_mean + 0.0 # 0.0 to make sure it's a new tensor
                if not use_mean_to_autoregress:
                    zt1_s += clipped_zt1_logstd.exp() * torch.randn_like(clipped_zt1_logstd) # Other places where sampling happens: citrisnf/lightning_module.py/get_losses
                z_t1_slice = torch.zeros_like(zt1_s)
                z_t1_slice[:, i] = zt1_s[:, i]
                pred_zt1 = pred_zt1 + z_t1_slice
        else:
            zt1_mean, zt1_logstd = self.prior_net(inp).chunk(2, dim=-1)
            zt1_s = zt1_mean
            if not self.hparams.sample_mean:
                zt1_s += zt1_logstd.exp() * torch.randn_like(zt1_logstd)
        sample = zt1_s if not return_mean_as_sample else zt1_mean
        if also_return_params:
            return sample, zt1_mean, zt1_logstd
        else:
            return sample


    @property
    def prior_parts(self):
        if self.hparams.autoregressive_prior:
            return [self.prior_net_init, self.prior_net_cond, self.prior_net_head]
        else:
            return [self.prior_net]

    def is_shared(self, param_name):
        if not self.hparams.per_latent_full_silos:
            return '3' not in param_name # The last layer, and the only one that contains completely specialized parameters
        else:
            return False

    def default_use_maxcorr_psi(self):
        return True

    def get_sparse_update_params(self):
        return ((n,p) for n, p in self.named_parameters() if 'prior' in n)

    def get_grouped_flp_params(self):
        groups = self.hparams.DataClass.SHORT_COARSE_FACTORS
        psi = self.get_learnt_or_maxcorr_psi(use_maxcorr=True)
        params = {'shared': []} | {f: [] for f in groups}
        for name, param in self.get_sparse_update_params():
            if '3.' in name: # hacky way to get the last layer
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