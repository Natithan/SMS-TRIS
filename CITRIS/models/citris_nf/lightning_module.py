"""
All models regarding CITRIS as PyTorch Lightning modules that we use for training.
"""
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F



from models.shared.flow_layers import NoOpFlow
import sys

from models.shared.triplet_evaluator import flow_based_triplet_pred

sys.path.append('../')
from models.shared import get_act_fn
from models.ae import Autoencoder
from models.citris_vae import CITRISVAE
from models.shared import AutoregNormalizingFlow
from util import tn, get_ckptpath_for_wid


class CITRISNF(CITRISVAE):
    """ 
    The main module implementing CITRIS-NF.
    It is a subclass of CITRIS-VAE to inherit several functionality.
    """

    def __init__(self, *args,
                        autoencoder_checkpoint=None,
                        num_flows=4,
                        hidden_per_var=16,
                        num_samples=8,
                        flow_act_fn='silu',
                        noise_level=-1,
                 skip_flow=False,
                 random_flow=False,
                 train_ae=False,
                 flow_wid=None,
                 freeze_flow=False,
                        **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs - see CITRIS-VAE for the full list
        autoencoder_checkpoint : str
                                 Path to the checkpoint of the autoencoder
                                 which should be used for training the flow
                                 on.
        num_flows : int
                    Number of flow layers to use
        hidden_per_var : int
                         Hidden dimensionality per latent variable to use
                         in the autoregressive networks.
        num_samples : int
                      Number of samples to take from an input encoding
                      during training. Larger sample sizes give smoother
                      gradients.
        flow_act_fn : str
                      Activation function to use in the networks of the flow
        noise_level : float
                      Standard deviation of the added noise to the encodings.
                      If smaller than zero, the std of the autoencoder is used.
        """
        kwargs['no_encoder_decoder'] = True  # We do not need any additional en- or decoder
        super().__init__(*args, **kwargs)
        assert not (skip_flow and random_flow), 'Cannot skip flow and use random flow at the same time'
        self.skip_flow = skip_flow
        self.random_flow = random_flow
        self.train_ae = train_ae
        if not self.skip_flow:
            # Initialize the flow
            if self.hparams.flow_wid is None:
                self.flow = AutoregNormalizingFlow(self.hparams.num_latents,
                                                   self.hparams.num_flows,
                                                   act_fn=get_act_fn(self.hparams.flow_act_fn),
                                                   hidden_per_var=self.hparams.hidden_per_var)
            else:
                assert not self.random_flow, 'Cannot use random flow and load a flow from a checkpoint at the same time'
                self.flow = AutoregNormalizingFlow.load_from_checkpoint(get_ckptpath_for_wid(flow_wid))
            if self.random_flow:
                # Ensure self.flow is not trainable
                for p in self.flow.parameters():
                    p.requires_grad_(False)
        else:
            self.flow = NoOpFlow()
        if freeze_flow:
            for p in self.flow.parameters():
                p.requires_grad_(False)
        # Setup autoencoder
        self.autoencoder = Autoencoder.load_from_checkpoint(self.hparams.autoencoder_checkpoint)
        if not self.train_ae:
            for p in self.autoencoder.parameters():
                p.requires_grad_(False)
        assert self.hparams.num_latents == self.autoencoder.hparams.num_latents, 'Autoencoder and flow need to have the same number of latents'
        assert self.autoencoder.hparams.img_width == self.causal_encoder.hparams.img_width, 'Autoencoder and Causal Encoder need to have the same image dimensions.'

        if self.hparams.noise_level < 0.0:
            self.hparams.noise_level = self.autoencoder.hparams.noise_level


    def encode(self, x_or_y, random=True):
        # Nathan: encode if raw images (instead of nonvariational-autoencoder-latents)
        if x_or_y.shape[-1] != self.hparams.num_latents:
            x = x_or_y
            y = self.autoencoder.encoder(x)
        else:
            y = x_or_y
        # Map input to disentangled latents, e.g. for correlation metrics
        if random:
            y = y + torch.randn_like(y) * self.hparams.noise_level
        z, ldj = self.flow(y)
        return z

    # Nathan-added
    def decode(self, z):
        y = self.flow.reverse(z)
        x = self.autoencoder.decoder(y)
        return x

    def sample_yt1(self, y_t, I_t1):
        '''
        Nathan: sample yt1 (the encoding after the frozen autoencoder, but before the flow network) given yt and I_t1
        '''
        z_t, ldj = self.flow(y_t)
        z_t1 = self.sample_zt1(z_t, I_t1)
        y_t1 = self.flow.reverse(z_t1)
        return y_t1

    # def sample_xt1(self, x_t, I_t1): # At the moment this half overlaps with sample_yt1
    #     z_t = self.encode(x_t,random=False)
    #     z_t1 = self.sample_zt1(z_t, I_t1)
    #     x_t1 = self.decode(z_t1)
    #     return x_t1

    def get_y_mseloss(self, batch):
        # yt, yt1, It1 = batch[0][:,0], batch[0][:,1], batch[1]
        yt, yt1, It1 = batch['encs'][:,0], batch['encs'][:,1], batch['targets']
        yt1_pred = self.sample_yt1(yt, It1)
        mseloss = F.mse_loss(yt1_pred, yt1)
        return mseloss

    def _get_loss(self, batch, mode='train'):
        """ Main training method for calculating the loss """
        target = batch['targets']
        if self.train_ae:
            raise NotImplementedError("Current implementation allows model to 'cheat' the objective. Need to implement in such a way that original images are used for the loss")
            # _, target, x = batch
            x = batch['imgs']
            # merge first two dimensions before feeding into autoencoder, then split again
            x_enc = self.autoencoder.encoder(x.reshape(-1, *x.shape[2:])).reshape(x.shape[0], x.shape[1], -1)
        else:
            # if len(batch) == 2:
            #     x_enc, target = batch
            # else:
            #     x_enc, _, target = batch
            x_enc = batch['encs']

        # Expand encodings over samples and add noise to 'sample' from the autoencoder
        # latent distribution
        x_enc = x_enc[...,None,:].expand(-1, -1, self.hparams.num_samples, -1) # (B, seq_len, Z) -> (B, seq_len, num_samples, Z). Aka x_enc forms the basis for num_samples samples
        batch_size, seq_len, num_samples, num_latents = x_enc.shape
        x_sample = x_enc + torch.randn_like(x_enc) * self.hparams.noise_level
        x_sample = x_sample.flatten(0, 2)
        # Execute the flow
        z_sample, ldj = self.flow(x_sample) # BLS_Z, BLS
        z_sample = z_sample.unflatten(0, (batch_size, seq_len, num_samples)) # BLS_Z -> B_L_S_Z
        # cut off too big (in absolute value) latent values
        z_sample_clamped = torch.clamp(z_sample, -10, 10)
        if self.hparams.cheat_cfd:
            latents = batch['lat']
            raise NotImplementedError
            # v_dict = self.z_to_c_predictor( # TODO finish

        ldj = ldj.reshape(batch_size, seq_len, num_samples) # BLS -> B_L_S
        # Calculate the negative log likelihood of the transition prior
        nll_z_sum = self.prior_t1.sample_based_nll(z_t=z_sample[:,:-1].flatten(0, 1),
                                             z_t1=z_sample[:,1:].flatten(0, 1),
                                             target=target.flatten(0, 1)) # B(L-1)
        # Add LDJ and prior NLL for full loss
        ldj = ldj[:,1:].flatten(0, 1).mean(dim=-1)  # Taking the mean over samples. B_L_S -> B_(L-1)_S -> B(L-1)_S -> B(L-1)
        nll_y_sum = nll_z_sum + ldj # B(L-1)
        loss = (nll_y_sum * self.hparams.beta_t1 * (seq_len - 1)).mean()
        # if not self.hparams.fullview_baseline:
        # Target classifier loss
        z_sample = z_sample.permute(0, 2, 1, 3).flatten(0, 1)  # Samples to batch dimension
        target = target[:,None].expand(-1, num_samples, -1, -1).flatten(0, 1)
        loss_model, loss_z = self.intv_classifier(z_sample=z_sample,
                                                  logger=self,
                                                  target=target,
                                                  transition_prior=self.prior_t1)
        loss = loss + (loss_model + loss_z) * self.hparams.beta_classifier

        # Logging
        self.log(f'{mode}_nll', nll_z_sum.mean())
        self.log(f'{mode}_ldj', ldj.mean())
        self.log(f'{mode}_intv_classifier_model', loss_model)
        self.log(f'{mode}_intv_classifier_z', loss_z)

        return loss


    def triplet_prediction(self, x_encs, source, use_cheating_target_assignment=False):
        """ Generates the triplet prediction of input encoding pairs and causal mask
        """
        # extracted to avoid code duplication with train_enc_cheat.py
        return flow_based_triplet_pred(self.autoencoder, self.flow, self.hparams.num_latents, source, self.get_maybe_cheating_target_assignment(use_cheating_target_assignment), x_encs)


    def add_teacher_forced_losses(self, encoder_outputs, It1, loss_dict, xtt1=None, ood_batch_size=0, args=None, grad_for_idloss=False, pretrain_phase=False):
        batch_size = xtt1.shape[0]
        # ldj, ztt1_sample = encoder_outputs['ldj'], encoder_outputs['ztt1_sample']
        ldj, ztt1_sample = [encoder_outputs[k].unflatten(0,(batch_size,-1)).movedim(2,1) for k in ['ldj', 'ztt1_sample']]
        seq_len = ztt1_sample.shape[1]
        # Calculate the negative log likelihood of the transition prior
        z_t = ztt1_sample[:, :-1].flatten(0, 1)  # B(L-1)_S0_Z
        z_t1 = ztt1_sample[:, 1:].flatten(0, 1)  # B(L-1)_S1_Z
        if pretrain_phase:
            overriden_fixed_logstd = False
        else:
            overriden_fixed_logstd = None
        nll_per_z, (prior_mean_per_hyp, prior_logstd_per_hyp) = \
            self.prior_t1.sample_based_nll(z_t=z_t,
                                           z_t1=z_t1,
                                           target=It1.flatten(0, 1),
                                           per_z=True,
                                           also_prior_params=True,
                                           overriden_fixed_logstd=overriden_fixed_logstd)  # B(L-1)_Z, (B(L-1)_S0_(C+1)_Z, B(L-1)_S0_(C+1)_Z)
        ldj_t1 = ldj[:, 1:] # B_L_S -> B_(L-1)_S
        # if ood_batch_size > 0 and args.coral_upweight:
        #     id_nll_per_z = nll_per_z[:-ood_batch_size].mean(0)
        #     ood_nll_per_z = nll_per_z[-ood_batch_size:].mean(0)
        #     nll_per_z = (id_nll_per_z + ood_nll_per_z)/2
        #     id_ldj_t1 = ldj_t1[:-ood_batch_size].flatten(0,1).mean()
        #     ood_ldj_t1 = ldj_t1[-ood_batch_size:].flatten(0,1).mean()
        #     ldj_t1_mean = (id_ldj_t1 + ood_ldj_t1)/2
        # else:
        #     nll_per_z = nll_per_z.mean(0)  # mean over batch dim. Z
        #     ldj_t1_mean = ldj_t1.flatten(0,1).mean()  # ldj: Taking the mean over samples and batch. B_(L-1)_S -> B(L-1)_S -> 1
        nll_per_z = nll_per_z.mean(0)  # mean over batch dim. Z
        ldj_t1_mean = ldj_t1.flatten(0,1).mean()  # ldj: Taking the mean over samples and batch. B_(L-1)_S -> B(L-1)_S -> 1
        loss_dict['nll_per_z'] = nll_per_z
        loss_dict['nll_z_sum'] = nll_per_z.sum()
        loss_dict['nll_y_sum'] = loss_dict['nll_z_sum'] + ldj_t1_mean
        loss_dict['fsl_train_loss'] = loss_dict['nll_y_sum'] # TODO not sure if this should include intv_classifier_losses, esp. when coral is used. For now NO during coral DA, YES during coral PT
        with (nullcontext() if grad_for_idloss else torch.no_grad()):
            loss_dict['intv_classifier_model'], loss_dict['intv_classifier_z'] = \
                self.intv_classifier(z_sample=ztt1_sample.permute(0, 2, 1, 3).flatten(0, 1),
                                     # Samples to batch dimension # B_L_S_Z -> B_S_L_Z -> BS_L_Z
                                     logger=self,
                                     target=It1[:, None].expand(-1, self.hparams.num_samples, -1, -1).flatten(0, 1),
                                     # B_L-1_C -> B_1_L-1_C -> B_S_L-1_C -> BS_L-1_C
                                     transition_prior=self.prior_t1)
        loss_dict['ID_loss'] = loss_dict['nll_y_sum'] * self.hparams.beta_t1 * (seq_len - 1) + \
                               (loss_dict['intv_classifier_model'] + loss_dict[
                                   'intv_classifier_z']) * self.hparams.beta_classifier

        return prior_logstd_per_hyp, prior_mean_per_hyp, z_t1

    def get_encoder_outputs(self, xtt1):
        batch_size, seq_len = xtt1.shape[0], xtt1.shape[1]
        ytt1 = self.autoencoder.encoder(xtt1.flatten(0, 1)).unflatten(0, (batch_size, seq_len))
        ytt1 = ytt1[..., None, :].expand(-1, -1, self.hparams.num_samples, -1)
        ytt1_flat = ytt1.flatten(0, 2)
        ytt1_sample_flat = (ytt1_flat + torch.randn_like(ytt1_flat) * self.hparams.noise_level)
        # ztt1_sample, ldj = [el.unflatten(0, (batch_size, seq_len, self.hparams.num_samples)) for el in
        #                     self.flow(ytt1_sample)]  # B_L_S_Z, B_L_S
        ztt1_sample, ldj = [el.unflatten(0, (batch_size, seq_len, self.hparams.num_samples)) for el in
                            self.flow(ytt1_sample_flat)]  # B_L_S_Z, B_L_S
        with torch.no_grad(): # Hopefully this is not too slow
            ztt1_mean, _ = [el.unflatten(0, (batch_size, seq_len, self.hparams.num_samples)) for el in
                            self.flow(ytt1_flat)]  # B_L_S_Z, B_L_S

        # flatten 'S' dimension into 'B' dimension
        ztt1_sample, ldj, ztt1_mean = [el.movedim(2, 1).flatten(0, 1) for el in [ztt1_sample, ldj, ztt1_mean]]  # B_L_S_Z -> B_S_L_Z -> BS_L_Z, B_L_S -> B_S_L -> BS_L, B_L_S_Z -> B_S_L_Z -> BS_L_Z

        return {"ldj": ldj, "ztt1_sample": ztt1_sample, "ztt1_mean": ztt1_mean}


    # def get_FSL_train_loss(self, batch):
    #     loss_dict = self.get_losses(batch)
    #     return loss_dict['fsl_train_loss']
    #     # return loss_dict['nll_z_mean']


    def log_mse_losses(self, batch, dataloader_idx, split='val'):
        super().log_mse_losses(batch, dataloader_idx, split)
        y_mseloss = self.get_y_mseloss(batch)
        maybe_cma_prefix = 'cma_' if  (dataloader_idx == 2) else ''
        self.mylog(f'{maybe_cma_prefix}{split}_y_mseloss', y_mseloss)

    @property
    def ae_part(self):
        return self.autoencoder

    def get_unexpanded_encoder_outputs(self, encoder_outputs, batch_size):
        ztt1_mean, ztt1_sample = encoder_outputs['ztt1_mean'], encoder_outputs['ztt1_sample']
        # BS_L_Z -> B_L_Z (indexing only the first sample)
        ztt1_mean, ztt1_sample = [el.unflatten(0, (batch_size, -1))[:, 0] for el in [ztt1_mean, ztt1_sample]]
        return ztt1_mean, ztt1_sample


    def get_encoder_outputs_for_tf(self, encoder_outputs, id_and_ood, id_batch_size):
        if id_and_ood:
            encoder_outputs_for_tf = {k: v[:id_batch_size*self.hparams.num_samples] for k, v in encoder_outputs.items()}
        else:
            encoder_outputs_for_tf = encoder_outputs
        return encoder_outputs_for_tf