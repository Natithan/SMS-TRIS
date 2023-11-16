import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
from util import tn
from pytorch_lightning.callbacks import LearningRateMonitor
# from data_generation.temporal_causal3dident.data_generation_causal3dident import ID_F2P, OOD_F2P

sys.path.append('../../')
from models.shared.utils import kl_divergence, gaussian_log_prob
from models.shared.modules import MultivarLinear, AutoregLinear, CosineWarmupScheduler, MaskedLinear, MultivarMaskedLinear


class AutoregressiveConditionalPrior(nn.Module):
    """
    The autoregressive base model for the autoregressive transition prior.
    The model is inspired by MADE and uses linear layers with masked weights.
    """

    def __init__(self, num_latents, num_blocks, c_hid, c_out, imperfect_interventions=True, fullview_baseline=False, no_init_masking=False, no_context_masking=False, per_latent_full_silos=False):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latent dimensions.
        num_blocks : int
                     Number of blocks to group the latent dimensions into. In other words,
                     it is the number of causal variables plus 1 (psi(0) - unintervened information).
        c_hid : int
                Hidden dimensionality to use in the network.
        c_out : int
                Output dimensionality per latent dimension (2 for Gaussian param estimation)
        imperfect_interventions : bool
                                  Whether interventions may be imperfect or not. If not, we mask the
                                  conditional information on interventions for a slightly faster
                                  convergence.
        fullview_baseline : bool
                            Whether to use the fullview baseline or not. (Nathan-added)
        """
        super().__init__()
        self.per_latent_full_silos = per_latent_full_silos
        no_interlatent_communication = self.per_latent_full_silos

        # Input layer for z_t
        self.context_layer = nn.Linear(num_latents, num_latents * c_hid)
        # Input layer for I_t
        self.target_layer = nn.Linear(num_blocks, num_latents * c_hid)
        # Autoregressive input layer for z_t+1
        self.init_layer = AutoregLinear(num_latents, inp_per_var=2, out_per_var=c_hid, diagonal=False, no_interlatent_communication=no_interlatent_communication)
        # Autoregressive main network with masked linear layers
        self.net = nn.Sequential(
            nn.SiLU(),
            AutoregLinear(num_latents, c_hid, c_hid, diagonal=True, no_interlatent_communication=no_interlatent_communication),
            nn.SiLU(),
            AutoregLinear(num_latents, c_hid, c_out, diagonal=True, no_interlatent_communication=no_interlatent_communication)
        )
        self.num_latents = num_latents
        self.imperfect_interventions = imperfect_interventions
        self.fullview_baseline = fullview_baseline
        self.no_init_masking = no_init_masking
        self.no_context_masking = no_context_masking
        self.register_buffer('target_mask', torch.eye(num_blocks))

    def forward(self, z_samples, z_previous, target_samples, target_true):
        """
        Given latent variables z^t+1, z^t, intervention targets I^t+1, and
        causal variable assignment samples from psi, estimate the prior
        parameters of p(z^t+1|z^t, I^t+1). This is done by running the
        autoregressive prior for each causal variable to estimate
        p(z_psi(i)^t+1|z^t, I_i^t+1), and stacking the i-dimension.


        Parameters
        ----------
        z_samples : torch.FloatTensor, shape [batch_size, num_latents]
                    The values of the latent variables at time step t+1, i.e. z^t+1.
        z_previous : torch.FloatTensor, shape [batch_size, num_latents]
                     The values of the latent variables at time step t, i.e. z^t.
        target_samples : torch.FloatTensor, shape [batch_size, num_latents, num_blocks]
                         The sampled one-hot vectors of psi for assigning latent variables to
                         causal variables.
        target_true : torch.FloatTensor, shape [batch_size, num_blocks]
                      The intervention target vector I^t+1.
        """

        num_total_factors = self.target_layer.weight.shape[1]
        num_nontrash_factors = num_total_factors - 1
        target_samples = target_samples.permute(0, 2, 1)  # shape: [B, num_total_factors, num_latents] or B_NTF_Z

        # region context_feats
        # Transform z^t into a feature vector.
        context_feats = self.context_layer(z_previous) # B_Z -> B_(Z*c_hid)
        # Expand over number of causal variables to run the prior i-times.
        context_feats = context_feats.unsqueeze(dim=1)  # shape: B_(Z*c_hid) -> B_1_(Z*c_hid)
        if not self.imperfect_interventions:
            # Mask out context features when having perfect interventions
            if not self.no_context_masking:
                context_feats = context_feats * (1 - target_true[..., None]) # B_1_(Z*c_hid) * B_NTF_1 -> B_NTF_(Z*c_hid)
        # endregion

        # region target_feats
        if not self.fullview_baseline:
            # Transform I^t+1 into feature vector, where only the i-th element is shown to the respective masked split.
            target_inp = target_true[:, None] * self.target_mask[None]  # B_1_NTF * NTF_NTF -> B_NTF_NTF. The second dimension varies for the psi-subgroup of zt+1 latents that will be predicted. The third dimension corresponds to causal factor for which intervention info is given.
            target_inp = target_inp - (1 - self.target_mask[None])  # Set -1 for masked values
        else:
            target_inp = (target_true - num_nontrash_factors) / num_total_factors  # This ensures that the effect is equivalent to a non-fullview baseline with psi = uniform

        target_feats = self.target_layer(target_inp) # B_NTF_(Z*c_hid)
        if self.fullview_baseline:
            # Expand over number of causal variables to run the prior i-times.
            target_feats = target_feats.unsqueeze(dim=1)  # shape: [batch_size, 1, num_latents * c_hid] B_1_(Z*c_hid)
        # endregion

        # region init_feats
        # Mask z^t+1 according to psi sample
        if self.no_init_masking:
            target_samples = torch.ones_like(target_samples)
        masked_samples = z_samples[:, None] * target_samples  # shape: [batch_size, num_blocks, num_latents] B_1_Z * B_NTF_Z -> B_NTF_Z
        masked_samples = torch.stack([masked_samples, target_samples * 2 - 1],
                                     dim=-1)  # shape: [batch_size, num_blocks, num_latents, 2] B_C_Z_2
        masked_samples = masked_samples.flatten(-2,
                                                -1)  # shape: [batch_size, num_blocks, 2*num_latents]. Interleaves the last dimension of z_samples*target_samples and target_samples.
        init_feats = self.init_layer(masked_samples) # B_NTF_(Z*c_hid)
        # endregion

        # Sum all features and use as input to feature network (division by 2 for normalization)
        feats = (
                            target_feats + init_feats + context_feats) / 2.0  # Nathan: what is the link between 2 and normalization?
        pred_params = self.net(
            feats)  # Nathan: the non-linear net doesn't maintain the equivalence of the fullview baseline with non-fullview-with-uniform-psi-samples. B_NTF_(Z*c_hid) -> B_NTF_(Z*c_out), c_out=2

        # Return prior parameters with first dimension stacking the different causal variables
        pred_params = pred_params.unflatten(-1, (
        self.num_latents, -1))  # shape: [batch_size, num_blocks, num_latents, c_out]
        return pred_params


class TransitionPrior(nn.Module):
    """
    The full transition prior promoting disentanglement of the latent variables across causal factors.
    """

    def __init__(self, num_latents, num_blocks, c_hid,
                 imperfect_interventions=False,
                 autoregressive_model=False,  # If true, masking happens
                 lambda_reg=0.01,
                 gumbel_temperature=1.0,
                 fullview_baseline=False,
                 std_min=0,
                 fixed_logstd=False,
                 no_init_masking=False,
                 no_context_masking=False,
                 per_latent_full_silos=False):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latent dimensions.
        num_blocks : int
                     Number of blocks to group the latent dimensions into. In other words,
                     it is the number of causal variables plus 1 (psi(0) - unintervened information).
        c_hid : int
                Hidden dimensionality to use in the prior network.
        imperfect_interventions : bool
                                  Whether interventions may be imperfect or not.
        autoregressive_model : bool
                               If True, an autoregressive prior model is used.
        lambda_reg : float
                     Regularizer for promoting intervention-independent information to be modeled
                     in psi(0)
        gumbel_temperature : float
                             Temperature to use for the Gumbel Softmax sampling.
        """
        super().__init__()
        self.num_latents = num_latents
        self.imperfect_interventions = imperfect_interventions
        self.gumbel_temperature = gumbel_temperature
        self.num_blocks = num_blocks
        self.autoregressive_model = autoregressive_model
        self.lambda_reg = lambda_reg
        self.std_min = std_min
        self.fixed_logstd = fixed_logstd
        self.fullview_baseline = fullview_baseline
        self.no_init_masking = no_init_masking
        self.no_context_masking = no_context_masking
        self.per_latent_full_silos = per_latent_full_silos
        assert self.lambda_reg >= 0 and self.lambda_reg < 1.0, 'Lambda regularizer must be between 0 and 1, excluding 1.'

        # Gumbel Softmax parameters of psi. Note that we model psi(0) in the last dimension for simpler implementation
        self.target_params = nn.Parameter(torch.zeros(num_latents, num_blocks + 1))
        if self.lambda_reg <= 0.0:  # No regularizer -> no reason to model psi(0)
            self.target_params.data[:, -1] = -9e15

        # For perfect interventions, we model the prior's parameters under intervention as a simple parameter vector here.
        if not self.imperfect_interventions:
            self.intv_prior = nn.Parameter(
                torch.zeros(num_latents, num_blocks, 2).uniform_(-0.5, 0.5))  # 2 for mean and std
        else:
            self.intv_prior = None

        # Prior model creation
        if autoregressive_model:
            self.prior_model = AutoregressiveConditionalPrior(num_latents, num_blocks + 1, 16, 2,
                                                              imperfect_interventions=self.imperfect_interventions,
                                                              fullview_baseline=self.fullview_baseline,
                                                              no_init_masking=self.no_init_masking,
                                                              no_context_masking=self.no_context_masking,
                                                                per_latent_full_silos=self.per_latent_full_silos)
        else:
            if self.per_latent_full_silos:
                raise NotImplementedError('Per-latent full silos not implemented for non-autoregressive models.')
            # Simple MLP per latent variable
            self.context_layer = nn.Linear(num_latents,
                                           c_hid * self.num_latents)  # mixes info from all latents. So is this where to check causality?
            self.inp_layer = MultivarLinear(1 + (self.num_blocks + 1 if self.imperfect_interventions else 0),
                                            c_hid, [self.num_latents])  # does not mix info from latents
            self.out_layer = nn.Sequential(
                nn.SiLU(),
                MultivarLinear(c_hid, c_hid, [self.num_latents]),
                nn.SiLU(),
                MultivarLinear(c_hid, 2,
                               [self.num_latents])
            )  # Does not (further) mix info from latents

            # Nathan
            if self.fullview_baseline:
                self.context_layer = nn.Linear(num_latents + num_blocks,
                                               c_hid * self.num_latents)

    def _get_prior_params(self, z_t, target=None, target_prod=None, target_samples=None, z_t1=None,overriden_fixed_logstd=None):
        """
        Abstracting the execution of the networks for estimating the prior parameters.

        Parameters
        ----------
        z_t : torch.FloatTensor, shape [batch_size, num_latents]
              Latents at time step t, i.e. the input to the prior
        target : torch.FloatTensor, shape [batch_size, num_blocks]
                 The intervention targets I^t+1
        target_prod : torch.FloatTensor, shape [batch_size, num_latents, num_blocks]
                      The true targets multiplied with the target sample mask, where masked
                      intervention targets are replaced with -1 to distinguish it from 0s.
        target_samples : torch.FloatTensor, shape [batch_size, num_latents, num_blocks]
                         The sampled one-hot vectors of psi for assigning latent variables to
                         causal variables.
        z_t1 : torch.FloatTensor, shape [batch_size, num_latents]
               Latents at time step t+1, i.e. the latents for which the prior parameters are estimated.
        """
        if self.autoregressive_model:
            prior_params = self.prior_model(z_samples=z_t1,
                                            z_previous=z_t,
                                            target_samples=target_samples,
                                            target_true=target)
            prior_params = prior_params.unbind(dim=-1)  # tensor -> tuple of tensors
        else:
            net_inp = z_t
            if self.fullview_baseline:
                context = self.context_layer(torch.cat([z_t, target[:, :-1]], dim=-1)).unflatten(-1,
                                                                                                 (self.num_latents, -1))
            else:
                context = self.context_layer(net_inp).unflatten(-1, (
                self.num_latents, -1))  # shape [batch_size, num_latents, c_hid] after unflattening

            net_inp_exp = net_inp.unflatten(-1, (
            self.num_latents, -1))  # = latents from previous time step, shape [batch_size, num_latents, 1]
            if self.imperfect_interventions:
                if target_prod is None:
                    target_prod = net_inp_exp.new_zeros(net_inp_exp.shape[:-1] + (self.num_blocks,))
                net_inp_exp = torch.cat([net_inp_exp, target_prod], dim=-1)
            block_inp = self.inp_layer(net_inp_exp)  # shape [batch_size, num_latents, c_hid]
            prior_params = self.out_layer(context + block_inp)  # shape [batch_size, num_latents, 2], 2 for mean and std
            prior_params = prior_params.chunk(2,
                                              dim=-1)  # shape tuple([batch_size, num_latents, 1],[batch_size, num_latents, 1])
            prior_params = [p.flatten(-2, -1) for p in
                            prior_params]  # shape tuple([batch_size, num_latents],[batch_size, num_latents])
        fixed_logstd = self.fixed_logstd if overriden_fixed_logstd is None else overriden_fixed_logstd
        if fixed_logstd and len(prior_params) == 2:
            prior_params = (prior_params[0], torch.zeros_like(prior_params[1]))
        return prior_params

    def kl_divergence(self, z_t, target, z_t1_mean, z_t1_logstd, z_t1_sample, also_prior_params=False, overriden_fixed_logstd=None):
        """
        Calculating the KL divergence between this prior's estimated parameters and
        the encoder on x^t+1 (CITRIS-VAE). Since this prior is in general much more
        computationally cheaper than the encoder/decoder, we marginalize the KL
        divergence over the target assignments for each latent where possible.

        Parameters
        ----------
        z_t : torch.FloatTensor, shape [batch_size, num_latents]
              Latents at time step t, i.e. the input to the prior
        target : torch.FloatTensor, shape [batch_size, num_blocks]
                 The intervention targets I^t+1
        z_t1_mean : torch.FloatTensor, shape [batch_size, num_latents]
                    The mean of the predicted Gaussian encoder(x^t+1)
        z_t1_logstd : torch.FloatTensor, shape [batch_size, num_latents]
                      The log-standard deviation of the predicted Gaussian encoder(x^t+1)
        z_t1_sample : torch.FloatTensor, shape [batch_size, num_latents]
                      A sample from the encoder distribution encoder(x^t+1), i.e. z^t+1
        """
        # region prep It1
        if len(target.shape) == 1:
            target_oh = F.one_hot(target, num_classes=self.num_blocks)
        else:
            target_oh = target
        # target_oh shape: [batch_size, num_blocks]

        # Sample a latent-to-causal assignment from psi
        target_probs = torch.softmax(self.target_params, dim=-1) # shape: [num_latents, num_blocks+1] = shape of psi
        target_samples = F.gumbel_softmax(self.target_params[None].expand(target.shape[0], -1, -1),
                                          tau=self.gumbel_temperature, hard=True)
        full_target_samples = target_samples  # full_target_samples shape: [batch_size, num_latents, num_blocks + 1]
        target_samples, no_target = target_samples[:, :, :-1], target_samples[:, :,
                                                               -1]  # target_samples shape: [batch_size, num_latents, num_blocks], no_target shape: [batch_size, num_latents]
        # Add I_0=0, i.e. no interventions on the noise/intervention-independent variables
        target_exp = torch.cat([target_oh, target_oh.new_zeros(target_oh.shape[0], 1)],
                               dim=-1)  # target_exp shape: [batch_size, num_blocks + 1]
        # endregion
        if self.autoregressive_model:
            # Run autoregressive model
            prior_params = self._get_prior_params(z_t, target_samples=full_target_samples, target=target_exp,
                                                  z_t1=z_t1_sample, overriden_fixed_logstd=overriden_fixed_logstd)
            kld_all = self._get_kld(z_t1_mean[:, None], z_t1_logstd[:, None], prior_params)
            # Regularize psi(0)
            if self.lambda_reg > 0.0:
                target_probs = torch.cat([target_probs[:, -1:], target_probs[:, :-1] * (1 - self.lambda_reg)], dim=-1)
            # Since to predict the parameters of z^t+1_i, we do not involve whether the target sample of i has been a certain value,
            # we can marginalize it over all possible target samples here.
            kld = (kld_all * target_probs.permute(1, 0)[None]).sum(dim=[1, 2])
        elif not self.imperfect_interventions:
            # For perfect interventions, we can estimate p(z^t+1|z^t) and p(z^t+1_j|I^t+1_i=1) independently.
            prior_params = self._get_prior_params(z_t, target_samples=full_target_samples, target=target_exp,
                                                  z_t1=z_t1_sample)
            kld_std = self._get_kld(z_t1_mean, z_t1_logstd, prior_params)
            intv_params = self._get_intv_params(z_t1_mean.shape, target=None)
            kld_intv = self._get_kld(z_t1_mean[..., None], z_t1_logstd[..., None], intv_params)

            target_probs, no_target_probs = target_probs[:, :-1], target_probs[:,
                                                                  -1]  # shapes: [num_latents, num_blocks], [num_latents]
            masked_intv_probs = (target_probs[None] * target_oh[:, None,
                                                      :])  # shape: [1,num_latents, num_blocks] x [batch_size, 1, num_blocks] -> [batch_size, num_latents, num_blocks]. [b,z,c] is masked if either Ib,c is 0 (no intervention), or psi_{z,c} is 0 (not assigned).
            intv_probs = masked_intv_probs.sum(
                dim=-1)  # Because we don't care which  intervention targets Ij are active for a z-dim that has some probability of being mapped to Ij, but just about the total probability of z being mapped to active Ij's.
            no_intv_probs = 1 - intv_probs - no_target_probs
            kld_intv_summed = (kld_intv * masked_intv_probs).sum(dim=-1)  # shape: [batch_size, num_latents]
            # Regularize by multiplying the KLD of psi(0) with (1-lambda_reg)
            kld = kld_intv_summed + kld_std * (no_intv_probs + no_target_probs * (1 - self.lambda_reg))
            kld = kld.sum(dim=-1)
        else:
            # For imperfect interventions, we estimate p(z^t+1_i|z^t,I^t+1_j) for all i,j, and marginalize over j.
            net_inps = z_t[:, None].expand(-1, self.num_blocks + 1, -1).flatten(0, 1)
            target_samples = torch.eye(self.num_blocks + 1, device=z_t.device)[None]
            target_oh = torch.cat([target_oh, target_oh.new_zeros(target_oh.shape[0], 1)], dim=-1)
            target_prod = target_oh[:, None] * target_samples - (1 - target_samples)
            target_prod = target_prod[:, :, None].expand(-1, -1, z_t.shape[-1], -1)
            target_prod = target_prod.flatten(0, 1)
            prior_params = self._get_prior_params(net_inps, target_prod=target_prod)
            prior_params = [p.unflatten(0, (-1, self.num_blocks + 1)) for p in prior_params]
            kld = self._get_kld(z_t1_mean[:, None], z_t1_logstd[:, None], prior_params)
            kld = (kld * target_probs.permute(1, 0)[None, :, :])
            kld = kld[..., :-1].sum(dim=[1, 2]) + (1 - self.lambda_reg) * kld[..., -1].sum(dim=1)

        if not also_prior_params:
            return kld
        else:
            return kld, prior_params

    def sample_based_nll(self, z_t, z_t1, target, per_z=False, also_prior_params=False,overriden_fixed_logstd=None):
        """
        Calculate the negative log likelihood of p(z^t1|z^t,I^t+1), meant for CITRIS-NF.
        We cannot make use of the KL divergence since the normalizing flow transforms
        the autoencoder distribution in a per-sample fashion. Nonetheless, to improve
        stability and smooth gradients, we allow z^t and z^t1 to be multiple samples
        for the same batch elements.

        Parameters
        ----------
        z_t : torch.FloatTensor, shape [batch_size, num_samples, num_latents]
              Latents at time step t, i.e. the input to the prior, with potentially
              multiple samples over the second dimension.
        z_t1 : torch.FloatTensor, shape [batch_size, num_samples, num_latents]
               Latents at time step t+1, i.e. the samples to estimate the prior for.
               Multiple samples again over the second dimension.
        target : torch.FloatTensor, shape [batch_size, num_blocks]
                 The intervention targets I^t+1
        """
        prior_mean, prior_logstd, target_probs = self.sample_based_prior_params(target, z_t, z_t1,overriden_fixed_logstd) # B_S0_(C+1)_Z, B_S0_(C+1)_Z, Z_(C+1)
        if self.autoregressive_model:
            # prior_mean - shape [batch_size, num_zt_samples, num_blocks, num_latents]
            # prior_logstd - shape [batch_size, num_zt_samples, num_blocks, num_latents]
            # z_t1 - shape [batch_size, num_zt1_samples, num_latents]
            nll = -gaussian_log_prob(prior_mean[:, :, None, :, :], prior_logstd[:, :, None, :, :], z_t1[:, None, :, None, :],std_minimum=self.std_min)  # B_S0_1_(C+1)_Z * B_1_S1_1_Z -> B_S0_S1_(C+1)_Z

            # We take the mean over samples, both over the z^t and z^t+1 samples.
            nll = nll.mean(dim=[1, 2])  # B_S0_S1_Z -> B_Z if fullview, else B_S0_S1_(C+1)_Z -> B_(C+1)_Z
            # Marginalize over target assignment
            nll = nll * target_probs.permute(1, 0)[None]  # B_(C+1)_Z * 1_(C+1)_Z -> B_(C+1)_Z
            # nll = nll.sum(dim=[1, 2])  # shape [batch_size]. B_(C+1)_Z -> B
            per_z_nll = nll.sum(dim=1)  # B_(C+1)_Z -> B_Z
            # nll = per_z_nll.sum(dim=-1)  # shape [batch_size]. B_Z -> B
        else:
            raise NotImplementedError("TODO check equivalence with OG CITRIS for non-autoregressive model")
            nll_std = -gaussian_log_prob(prior_mean[..., None, :], prior_logstd[..., None, :], z_t1[..., None, :, :],std_minimum=self.std_min)
            per_z_nll_std = nll_std.mean(dim=[1, 2])  # Averaging over input and output samples
            if not self.imperfect_interventions:

                if not self.fullview_baseline:
                    # intv_params is a separate set of parameters that predicts each z_{t+1,j} in case the I_{t+1}[i] to which psi (target_probs) assigns z_{t+1,j} is 1.
                    # It then predicts the z_{t+1,j} purely based on I_{t+1}[i] and NOT z_t.
                    intv_params = self._get_intv_params(z_t.shape, target=None)

                    intv_mean, intv_logstd = intv_params[0], intv_params[1] # B_Z_C
                    nll_intv = -gaussian_log_prob(intv_mean[:, None, :, :].detach(),
                                                  intv_logstd[:, None, :, :].detach(),
                                                  z_t1[..., None],std_minimum=self.std_min) # B_1_Z_C * B_S0_Z_1 -> B_S0_Z_C
                    nll_intv = nll_intv.mean(dim=1)  # Averaging over output samples (input all the same). B_S0_Z_C -> B_Z_C. nll_intv - shape: [batch_size, num_latents, num_blocks]

                    target_probs, no_target_probs = target_probs[:, :-1], target_probs[:, -1] # Z_C, Z
                    masked_intv_probs = (
                                target_probs[None] * target_oh[:, None, :])  # 1_Z_C * B_1_C -> B_Z_C

                    # This contains the contribution to the total probability of a z-dimension coming from masked_intv_probs, =
                    # prior where [z belongs to a certain I (psi aka target_probs) * whether that I is 1 for this sample (target_oh)] *
                    # the nll of z under that prior.
                    per_z_nll_intv = (nll_intv * masked_intv_probs).sum(dim=-1)  # B_Z_C * B_Z_C -> B_Z

                    # So if intv_probs is eg 0 (due to psi allocating zero probability of z belonging to an intervened I, or there being no intervened I),
                    # then no_intv_probs is 1 - no_target_probs, which is the probability psi allocates to [z belonging to no intervened I].
                    # if intv_probs is close to 1, then no_target_probs is close to 0, and this is close to zero.
                    # So it combines the contributions to the total probability of hypotheses where z is mapped by psi to no I, or the I z is mapped to is 0.
                    intv_probs = masked_intv_probs.sum(dim=-1)  # B_Z_C -> B_Z
                    no_intv_probs = 1 - intv_probs - no_target_probs * (self.lambda_reg) # B_Z

                    # per_z_nll_std is the nll under the hypothesis that z belongs to an untintervened I, or never-intervened I.
                    per_z_nll = per_z_nll_intv + per_z_nll_std * no_intv_probs
                else:
                    per_z_nll = per_z_nll_std
                # nll = per_z_nll.sum(dim=-1)

                if not self.fullview_baseline:
                    # Nathan: is this just a way to double the size of the gradient for self.intv_priors?
                    nll_intv_noz = -gaussian_log_prob(intv_mean[:, None, :, :], # B_1_Z_C
                                                      intv_logstd[:, None, :, :], # B_1_Z_C
                                                      z_t1[..., None].detach(),std_minimum=self.std_min) # B_S0_Z_1 --> B_S0_Z_C
                    nll_intv_noz = nll_intv_noz.mean(dim=1) # B_S0_Z_C -> B_Z_C
                    # nll_intv_noz = (nll_intv_noz * target_oh[:, None, :]).sum(dim=[1, 2]) # B_Z_C * B_1_C -> B_Z_C -> B
                    # nll = nll + nll_intv_noz - nll_intv_noz.detach()  # No noticable increase in loss, only grads
                    per_z_nll_intv_noz = (nll_intv_noz * target_oh[:, None, :]).sum(dim=-1) # B_Z_C * B_1_C -> B_Z_C -> B_Z
                    per_z_nll = per_z_nll + per_z_nll_intv_noz - per_z_nll_intv_noz.detach() # B_Z
                    # nll = per_z_nll.sum(dim=-1) # B_Z -> B
            else:
                per_z_nll = nll_std  # Averaging over input and output samples
                # nll = per_z_nll.sum(dim=-1)
        if not self.fullview_baseline:  # Psi is not relevant for the fullview baseline
            # In Normalizing Flows, we do not have a KL divergence to easily regularize with lambda_reg.
            # Instead, we regularize the Gumbel Softmax parameters to maximize the probability for psi(0).
            if self.lambda_reg > 0.0 and self.training:
                target_params_soft = torch.softmax(self.target_params, dim=-1) # Z_C+1
                # nll = nll + self.lambda_reg * (1 - target_params_soft[:, -1]).mean(dim=0)
                per_z_nll = per_z_nll + (self.lambda_reg * (1 - target_params_soft[:, -1])[None,:]) / per_z_nll.shape[-1] # B_Z
                # nll = per_z_nll.sum(dim=-1) # B_Z -> B
        nll = per_z_nll.sum(dim=-1)  # B_Z -> B
        if not per_z:
            if also_prior_params:
                return nll, (prior_mean, prior_logstd)
            else:
                return nll
        else:
            if also_prior_params:
                return per_z_nll, (prior_mean, prior_logstd)
            else:
                return per_z_nll

    def sample_based_prior_params(self, target, z_t, z_t1,overriden_fixed_logstd=None):
        # Only for self.autoregressive==True model does this give final prior params.
        # In other cases, the prior params need to be further processed by the code in sample_based_nll.
        batch_size, num_samples, _ = z_t.shape
        target_exp, target_oh, target_probs, target_prod, target_samples = self.prep_target_derivates_for_prior(
            batch_size, num_samples, target)
        # Obtain estimated prior parameters for p(z^t1|z^t,I^t+1)
        # B*S0 and B*S1 are used as the same here.
        prior_params = self._get_prior_params(z_t.flatten(0, 1),  # B*S0_Z
                                              target_samples=target_samples,  # B*S1_Z_(C+1)
                                              target=target_exp,  # B*S1_(C+1)
                                              target_prod=target_prod,  # B*S1_Z_(C+1)
                                              z_t1=z_t1.flatten(0, 1),
                                              overriden_fixed_logstd=overriden_fixed_logstd)  # B*S1_Z
        prior_mean, prior_logstd = [p.unflatten(0, (batch_size, num_samples)) for p in prior_params]
        return prior_mean, prior_logstd, target_probs

    def prep_target_derivates_for_prior(self, batch_size, num_samples, target):
        # region Prep It1
        if len(target.shape) == 1:
            target_oh = F.one_hot(target, num_classes=self.num_blocks)
        else:
            target_oh = target
        # target_oh.shape: [batch_size, num_blocks]

        # Sample a latent-to-causal assignment from psi
        target_probs = torch.softmax(self.target_params, dim=-1)  # shape: [num_latents, num_blocks+1] = shape of psi
        target_samples = F.gumbel_softmax(self.target_params[None].expand(batch_size * num_samples, -1, -1),
                                          tau=self.gumbel_temperature,
                                          hard=True)  # shape: [batch_size*num_samples, num_latents, num_blocks+1]
        # Add sample dimension and I_0=0 to the targets
        target_exp = target_oh[:, None].expand(-1, num_samples, -1).flatten(0,
                                                                            1)  # shape: [batch_size*num_samples, num_blocks]
        target_exp = torch.cat([target_exp, target_exp.new_zeros(batch_size * num_samples, 1)],
                               dim=-1)  # shape: [batch_size*num_samples, num_blocks+1]; adding the trash dimension
        target_prod = target_exp[:, None, :] * target_samples - (
                1 - target_samples)  # (B*S1)_1_(C+1) * (B*S1)_Z_(C+1) -> (B*S1)_Z_(C+1). z_{psi_{i}} for which I_i=1 -> 1, z_{psi_{i}} for which I_i=0 -> 0, z_{psi_{i}} for which I_j=whatever, with i != j, -> -1
        # endregion
        return target_exp, target_oh, target_probs, target_prod, target_samples

    def _get_kld(self, true_mean, true_logstd, prior_params):
        # Function for cleaning up KL divergence calls
        kld = kl_divergence(true_mean, true_logstd, prior_params[0], prior_params[1])
        return kld

    def _get_intv_params(self, shape, target):
        # Return the prior parameters for p(z^t+1_j|I^t+1_i=1)
        intv_params = self.intv_prior[None].expand(shape[0], -1, -1, -1)
        if target is not None:
            intv_params = (intv_params * target[:, None, :, None]).sum(dim=2)
        return intv_params[..., 0], intv_params[..., 1]

    def get_target_assignment(self, hard=False):
        # Returns psi, either 'hard' (one-hot, e.g. for triplet eval) or 'soft' (probabilities, e.g. for debug)
        if not hard:
            return torch.softmax(self.target_params, dim=-1)
        else:
            return F.one_hot(torch.argmax(self.target_params, dim=-1), num_classes=self.target_params.shape[-1])

# class TrueLatentPrior(pl.LightningModule):
#     """
#     Transition prior that uses true latents C instead of autoencoder-produced Z.
#     """
#
#     def __init__(self, num_causal_vars, c_hid, c_out=2, true_ood_shifts=None, only_parents=True,
#                  nonfv_target_layer=False, **kwargs):
#         """
#         Parameters
#         ----------
#         c_hid : int
#                 Hidden dimensionality to use in the network.
#         c_out : int
#                 Output dimensionality per latent dimension (2 for Gaussian param estimation)
#         """
#         super().__init__()
#         if true_ood_shifts is None:
#             true_ood_shifts = []
#         self.num_coarse_vars = num_causal_vars #              pos,                 rot,                 rot-spot, hue-object, hue-spot, hue-back, obj-shape
#         if kwargs['data_folder'] == 'causal3d_time_dep_all7_conthue_01_coarse':
#             self.num_fine_vars = self.num_coarse_vars + 3 #   pos-x, pos-y, pos-z, rot-alpha, rot-beta, rot-spot, hue-object, hue-spot, hue-back, obj-shape
#             self.fine2coarse = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
#         else:
#             print(f'Only causal3d_time_dep_all7_conthue_01_coarse is supported for now, but got {kwargs["data_folder"]}')
#             raise NotImplementedError
#         self.save_hyperparameters()
#         self.old = kwargs['old'] # TODO the not old version is worse for some reason
#         # region context layer
#         if only_parents:
#             self.factor_id_to_parent_ids = {FACTORS.index(k): [FACTORS.index(v) for v in vlist] for k, vlist in ID_F2P.items()}
#             for factor in true_ood_shifts:
#                 long_factor = SHORT2F[factor]
#                 self.factor_id_to_parent_ids[FACTORS.index(long_factor)] = [FACTORS.index(v) for v in OOD_F2P[long_factor]]
#             if self.old:
#                 true_parent_mask = torch.zeros(self.num_fine_vars * c_hid, self.num_fine_vars, dtype=torch.bool)
#                 # Set parent matches to one in true_parent_mask
#                 for factor_id, parent_ids in self.factor_id_to_parent_ids.items():
#                     true_parent_mask[factor_id * c_hid:(factor_id + 1) * c_hid, parent_ids] = True
#                 self.context_layer = MaskedLinear(self.num_fine_vars, self.num_fine_vars * c_hid, mask=true_parent_mask)
#             else:
#                 true_parent_mask = torch.zeros(self.num_fine_vars, c_hid, self.num_fine_vars, dtype=torch.bool)
#                 # Set parent matches to one in true_parent_mask
#                 for factor_id, parent_ids in self.factor_id_to_parent_ids.items():
#                     true_parent_mask[factor_id, :, parent_ids] = True
#                 self.context_layer = MultivarMaskedLinear(input_dims=self.num_fine_vars, output_dims=c_hid, extra_dims=[self.num_fine_vars], mask=true_parent_mask, nokaiming=kwargs['nokaiming'])
#         else:
#             if self.old:
#                 self.context_layer = nn.Linear(self.num_fine_vars, self.num_fine_vars * c_hid)
#             else:
#                 self.context_layer = MultivarLinear(input_dims=self.num_fine_vars, output_dims=c_hid, extra_dims=[self.num_fine_vars], nokaiming=kwargs['nokaiming'])
#         # endregion
#         # region target layer
#         if nonfv_target_layer:
#             if self.old:
#                 target_mask = torch.zeros(self.num_fine_vars * c_hid, self.num_coarse_vars, dtype=torch.bool)
#                 # Set blocks of elements on the diagonal to one in target_mask
#                 for factor_id in range(self.num_fine_vars):
#                     target_mask[factor_id * c_hid:(factor_id + 1) * c_hid, self.fine2coarse[factor_id]] = True
#                 self.target_layer = MaskedLinear(self.num_coarse_vars, self.num_fine_vars * c_hid, mask=target_mask)
#             else:
#                 target_mask = torch.zeros(self.num_fine_vars, c_hid, self.num_coarse_vars, dtype=torch.bool)
#                 # Set blocks of elements on the diagonal to one in target_mask
#                 for factor_id in range(self.num_fine_vars):
#                     target_mask[factor_id, :, self.fine2coarse[factor_id]] = True
#                 self.target_layer = MultivarMaskedLinear(input_dims=self.num_coarse_vars, output_dims=c_hid, extra_dims=[self.num_fine_vars], mask=target_mask, nokaiming=kwargs['nokaiming'])
#         else:
#             if self.old:
#                 self.target_layer = nn.Linear(self.num_coarse_vars, self.num_fine_vars * c_hid)
#             else:
#                 self.target_layer = MultivarLinear(input_dims=self.num_coarse_vars, output_dims=c_hid, extra_dims=[self.num_fine_vars], nokaiming=kwargs['nokaiming'])
#         # endregion
#
#         self.net = nn.Sequential(
#             nn.SiLU(),
#             MultivarLinear(input_dims=c_hid, output_dims=c_hid, extra_dims=[self.num_fine_vars]),
#             nn.SiLU(),
#             MultivarLinear(input_dims=c_hid, output_dims=c_out, extra_dims=[self.num_fine_vars])
#         )
#         # print("REMOVE THIS AFTER EXPERIMENT" + "=" * 100)
#         # if self.old:
#         #     torch.save(self, 'old_model.ckpt')
#         # else:
#         #     from experiments.check_multivar_TLP import match_MV_to_nonMV
#         #     old_model = torch.load('old_model.ckpt')
#         #     match_MV_to_nonMV(self, old_model)
#         #     print("Matched MV to nonMV")
#         #     del old_model
#
#     def training_step(self, batch, batch_idx):
#         loss_dict = self.get_losses(batch)
#         self.log_losses(loss_dict, 'train')
#         return loss_dict['loss']
#
#     def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
#         if not 'context_layer.mask' in checkpoint['state_dict']: # For backwards compatibility
#             self.context_layer = nn.Linear(self.num_fine_vars, self.num_fine_vars * self.hparams.c_hid)
#
#     def get_loss(self, batch):
#         return self.get_losses(batch)['loss']
#
#     def get_losses(self, batch):
#         loss_dict = {}
#         loss_dict['per_c_loss'] = self.get_per_c_loss(batch)
#         loss_dict['loss'] = loss_dict['per_c_loss'].mean()
#         return loss_dict
#
#     def get_per_c_loss(self, batch):
#         Ct, Ct1, It1 = self.unpack_batch(batch)
#         Ct1_pred = self(Ct, It1)
#         nll = -gaussian_log_prob(Ct1_pred[..., 0], Ct1_pred[..., 1], Ct1)
#         per_c_loss = nll.mean(dim=0)
#         return per_c_loss
#
#     def unpack_batch(self, batch):
#         _, It1, Ctt1 = batch
#         It1 = It1.squeeze(1).float()
#         Ct, Ct1 = Ctt1[:, 0], Ctt1[:, 1]
#         return Ct, Ct1, It1
#
#     def validation_step(self, batch, batch_idx):
#         loss_dict = self.get_losses(batch)
#         self.log_losses(loss_dict, 'val')
#         return loss_dict['loss']
#
#     def log_losses(self, loss_dict, split):
#         self.log(f'{split}_loss', loss_dict['loss'])
#         for i, c_loss in enumerate(loss_dict['per_c_loss']):
#             self.log(f'{split}_loss_{FACTORS[i]}', c_loss)
#
#     def test_step(self, batch, batch_idx):
#         loss_dict = self.get_losses(batch)
#         self.log_losses(loss_dict, 'test')
#         return loss_dict['loss']
#
#     def forward(self, Ct, It1):
#
#         # region context_feats
#         # Transform C^t into a feature vector.
#         context_feats = self.context_layer(Ct)
#         # endregion
#
#         # region target_feats
#         target_feats = self.target_layer(It1)
#         # endregion
#
#         # Sum all features and use as input to feature network (division by 2 for normalization)
#         feats = (target_feats + context_feats) / 2.0
#         if self.old:
#             feats = feats.unflatten(-1,(self.num_fine_vars,-1))
#         pred_params_parallel = self.net(
#             feats)
#         return pred_params_parallel
#
#     @staticmethod
#     def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, correlation_dataset=None, correlation_test_dataset=None, mini=False,**kwargs):
#         callbacks = []
#         # Create learning rate callback
#         callbacks.append(LearningRateMonitor('step'))
#         return callbacks
#
#     def configure_optimizers(self):
#
#         optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
#         lr_scheduler = CosineWarmupScheduler(optimizer,
#                                              warmup=self.hparams.warmup,
#                                              max_iters=self.hparams.max_iters)
#         return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
#
#     def get_masks_for_unfreeze(self, unfreeze_idx):
#         mask_list = []
#         for param in self.parameters():
#             param_mask = torch.zeros_like(param)
#             param_mask[unfreeze_idx] = 1
#             mask_list.append(param_mask)
#         return mask_list

class OldMixedTrueLatentPrior(pl.LightningModule):
    '''
    Explicitly mixes dimensions of the trueLatentPrior in some fixed, way,
    with the idea that this makes aligning mechanisms with parameters impossible, unless mixed by (something close
    to) the identity matrix.
    '''
    def __init__(self, num_causal_vars, c_hid, c_out=2, true_ood_shifts=None, only_parents=True,
                 nonfv_target_layer=False, mix=False, **kwargs):
        """
        Parameters
        ----------
        c_hid : int
                Hidden dimensionality to use in the network.
        c_out : int
                Output dimensionality per latent dimension (2 for Gaussian param estimation)
        """
        super().__init__()
        self.num_coarse_vars = num_causal_vars #              pos,                 rot,                 rot-spot, hue-object, hue-spot, hue-back, obj-shape
        if kwargs['data_folder'] == 'causal3d_time_dep_all7_conthue_01_coarse':
            self.num_fine_vars = self.num_coarse_vars + 3 #   pos-x, pos-y, pos-z, rot-alpha, rot-beta, rot-spot, hue-object, hue-spot, hue-back, obj-shape
            self.fine2coarse = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
        else:
            print(f'Only causal3d_time_dep_all7_conthue_01_coarse is supported for now, but got {kwargs["data_folder"]}')
            raise NotImplementedError
        self.save_hyperparameters()
        self.old = kwargs['old'] # TODO the not old version is worse for some reason
        print("Warning: refactored this without checking, since I'm assuming I won't use this class anymore")
        self.DA_loss_factor_keys = ['c_mean', 'c_mx_mean'] + ['c_' + el for el in SHORT_FACTORS_old] + [f'c_mx_{i}' for i in enumerate(SHORT_FACTORS_old)]

        # region context layer
        if only_parents and not mix:
            self.factor_id_to_parent_ids = {FACTORS_old.index(k): [FACTORS_old.index(v) for v in vlist] for k, vlist in ID_F2P.items()}
            if true_ood_shifts is None:
                true_ood_shifts = []
            for factor in true_ood_shifts:
                long_factor = SHORT2F_old[factor]
                self.factor_id_to_parent_ids[FACTORS_old.index(long_factor)] = [FACTORS_old.index(v) for v in OOD_F2P[long_factor]]
            if self.old:
                true_parent_mask = torch.zeros(self.num_fine_vars * c_hid, self.num_fine_vars, dtype=torch.bool)
                # Set parent matches to one in true_parent_mask
                for factor_id, parent_ids in self.factor_id_to_parent_ids.items():
                    true_parent_mask[factor_id * c_hid:(factor_id + 1) * c_hid, parent_ids] = True
                self.context_layer = MaskedLinear(self.num_fine_vars, self.num_fine_vars * c_hid, mask=true_parent_mask)
            else:
                true_parent_mask = torch.zeros(self.num_fine_vars, c_hid, self.num_fine_vars, dtype=torch.bool)
                # Set parent matches to one in true_parent_mask
                for factor_id, parent_ids in self.factor_id_to_parent_ids.items():
                    true_parent_mask[factor_id, :, parent_ids] = True
                self.context_layer = MultivarMaskedLinear(input_dims=self.num_fine_vars, output_dims=c_hid, extra_dims=[self.num_fine_vars], mask=true_parent_mask, nokaiming=kwargs['nokaiming'])
        else:
            if self.old:
                self.context_layer = nn.Linear(self.num_fine_vars, self.num_fine_vars * c_hid)
            else:
                self.context_layer = MultivarLinear(input_dims=self.num_fine_vars, output_dims=c_hid, extra_dims=[self.num_fine_vars], nokaiming=kwargs['nokaiming'])
        # endregion
        # region target layer
        if nonfv_target_layer:
            if self.old:
                target_mask = torch.zeros(self.num_fine_vars * c_hid, self.num_coarse_vars, dtype=torch.bool)
                # Set blocks of elements on the diagonal to one in target_mask
                for factor_id in range(self.num_fine_vars):
                    target_mask[factor_id * c_hid:(factor_id + 1) * c_hid, self.fine2coarse[factor_id]] = True
                self.target_layer = MaskedLinear(self.num_coarse_vars, self.num_fine_vars * c_hid, mask=target_mask)
            else:
                target_mask = torch.zeros(self.num_fine_vars, c_hid, self.num_coarse_vars, dtype=torch.bool)
                # Set blocks of elements on the diagonal to one in target_mask
                for factor_id in range(self.num_fine_vars):
                    target_mask[factor_id, :, self.fine2coarse[factor_id]] = True
                self.target_layer = MultivarMaskedLinear(input_dims=self.num_coarse_vars, output_dims=c_hid, extra_dims=[self.num_fine_vars], mask=target_mask, nokaiming=kwargs['nokaiming'])
        else:
            if self.old:
                self.target_layer = nn.Linear(self.num_coarse_vars, self.num_fine_vars * c_hid)
            else:
                self.target_layer = MultivarLinear(input_dims=self.num_coarse_vars, output_dims=c_hid, extra_dims=[self.num_fine_vars], nokaiming=kwargs['nokaiming'])
        # endregion
        # region net
        self.net = nn.Sequential(
            nn.SiLU(),
            MultivarLinear(input_dims=c_hid, output_dims=c_hid, extra_dims=[self.num_fine_vars]),
            nn.SiLU(),
            MultivarLinear(input_dims=c_hid, output_dims=c_out, extra_dims=[self.num_fine_vars])
        )
        # endregion
        # Non-trainable mixing matrix of self.num_fine_var x self.num_fine_var.

        if mix:
            mixing_matrix = torch.randn(self.num_fine_vars, self.num_fine_vars)
            # Make it orthogonal so that it is easily invertible. Based on https://discuss.pytorch.org/t/how-to-efficiently-and-randomly-produce-an-orthonormal-matrix/153609/2
            mixing_matrix = torch.linalg.qr(mixing_matrix)[0]
            unmixing_matrix = mixing_matrix.transpose(0,1)

        else:
            mixing_matrix = torch.eye(self.num_fine_vars)
            unmixing_matrix = torch.eye(self.num_fine_vars)
        self.register_buffer('mixing_matrix', mixing_matrix)
        self.register_buffer('unmixing_matrix', unmixing_matrix)

    def training_step(self, batch, batch_idx):
        loss_dict = self.get_losses(batch)
        self.log_losses(loss_dict, 'train')
        return loss_dict['nll_c_mx_mean']

    def test_step(self, batch, batch_idx):
        loss_dict = self.get_losses(batch)
        self.log_losses(loss_dict, 'test')
        return loss_dict['nll_c_mx_mean']

    def validation_step(self, batch, batch_idx):
        loss_dict = self.get_losses(batch)
        self.log_losses(loss_dict, 'val')
        return loss_dict['nll_c_mx_mean']


    # def get_FSL_train_loss(self, batch):
    #     return self.get_losses(batch)['nll_c_mx_mean']


    def get_losses_(self, batch): #, also_mae=False, also_logstd=False):
        Ct_mixed, Ct1_mixed, It1 = self.unpack_and_mix_batch(batch)
        Ct1_mixed_pred = self(Ct_mixed, It1)
        Ct1_mixed_pred_m, Ct1_mixed_pred_std = Ct1_mixed_pred[..., 0], Ct1_mixed_pred[..., 1]
        nll_mixed_per_c = -gaussian_log_prob(Ct1_mixed_pred_m, Ct1_mixed_pred_std, Ct1_mixed,std_minimum=self.std_min).mean(dim=0)

        nll_unmixed_per_c = -gaussian_log_prob(*[el @ self.unmixing_matrix for el in (Ct1_mixed_pred_m, Ct1_mixed_pred_std, Ct1_mixed)],std_minimum=self.std_min).mean(dim=0)
        loss_dict = {
            'nll_c_mx_mean': nll_mixed_per_c.mean(),
            'nll_c_mean': nll_unmixed_per_c.mean(),
            'nll_per_c_mx': nll_mixed_per_c,
            'nll_per_c': nll_unmixed_per_c,
        }
        # if also_mae:
            # loss_dict['mse_mixed_per_c'] = F.mse_loss(Ct1_mixed_pred_m, Ct1_mixed, reduction='none').mean(dim=0)
            # loss_dict['mse_unmixed_per_c'] = F.mse_loss(Ct1_mixed_pred_m @ self.unmixing_matrix, Ct1_mixed @ self.unmixing_matrix, reduction='none').mean(dim=0)
        loss_dict['mae_per_c_mx'] = abs(Ct1_mixed_pred_m - Ct1_mixed).mean(dim=0)
        loss_dict['mae_per_c'] = abs(Ct1_mixed_pred_m @ self.unmixing_matrix - Ct1_mixed @ self.unmixing_matrix).mean(dim=0)
        loss_dict['mae_c_mx_mean'] = loss_dict['mae_per_c_mx'].mean()
        loss_dict['mae_c_mean'] = loss_dict['mae_per_c'].mean()
        # if also_logstd:
        loss_dict['logstd_per_c_mx'] = Ct1_mixed_pred_std.mean(dim=0)
        loss_dict['logstd_per_c'] = (Ct1_mixed_pred_std @ self.unmixing_matrix).mean(dim=0)
        loss_dict['logstd_c_mx_mean'] = loss_dict['logstd_per_c_mx'].mean()
        loss_dict['logstd_c_mean'] = loss_dict['logstd_per_c'].mean()
        return loss_dict

    def log_losses(self, loss_dict, split):
        # TODO this should be updates with correct loss keys. Not sure if I'm going to use this model again, so postponing until I do.
        self.log(f'{split}_loss_unmixed_mean', loss_dict['loss_unmixed_mean'])
        self.log(f'{split}_loss_mixed_mean', loss_dict['loss_mixed_mean'])
        for i, c_loss in enumerate(loss_dict['loss_unmixed_per_c']):
            self.log(f'{split}_loss_{FACTORS_old[i]}', c_loss)
        for i, c_loss in enumerate(loss_dict['loss_mixed_per_c']):
            self.log(f'{split}_loss_mixed_{i}', c_loss)


    def unpack_and_mix_batch(self, batch):
        _, It1, Ctt1 = batch
        It1 = It1.squeeze(1).float()
        Ct, Ct1 = Ctt1[:, 0], Ctt1[:, 1]
        if self.hparams.mix:
            Ct_mixed, Ct1_mixed = Ct @ self.mixing_matrix, Ct1 @ self.mixing_matrix
            return Ct_mixed, Ct1_mixed, It1
        else:
            return Ct, Ct1, It1

    def forward(self, Ct, It1):

        # region context_feats
        # Transform C^t into a feature vector.
        context_feats = self.context_layer(Ct)
        # endregion

        # region target_feats
        target_feats = self.target_layer(It1)
        # endregion

        # Sum all features and use as input to feature network (division by 2 for normalization)
        feats = (target_feats + context_feats) / 2.0
        if self.old:
            feats = feats.unflatten(-1,(self.num_fine_vars,-1))
        pred_params_parallel = self.net(
            feats)
        return pred_params_parallel

    @staticmethod
    def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, correlation_dataset=None, correlation_test_dataset=None, mini=False,**kwargs):
        callbacks = []
        # Create learning rate callback
        callbacks.append(LearningRateMonitor('step'))
        return callbacks

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]




