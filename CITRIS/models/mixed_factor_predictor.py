from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Function
import torch.nn.functional as F

from experiments.utils import coral_loss, maybe_id_and_ood_unpack
from models.shared.causal_encoder import get_causal_encoder, CausalEncoder
from models.shared.next_step_predictor import NextStepPredictor
from util import tn
import pytorch_lightning as pl
from models.shared.callbacks import LearningRateMonitor, CorrelationMetricsLogCallback, expand_per_group, \
    BaselineCorrelationMetricsLogCallback
from models.shared import MultivarLinear, gaussian_log_prob, CosineWarmupScheduler


# class MixedFactorPredictor(pl.LightningModule):
class MixedFactorPredictor(NextStepPredictor):
    '''
    This model aims to predict future synthetic mix d=mix(c) of the TRIS-data-true-latents c.
    It consists of an encoder that produces an e that may or may not be aligned with c.
    There is an option to replace the trainable encoder with a non-trainable encoder that perfectly unmixes c and thus aligns e with d.
    The intent is to compare a model with a trained encoder to one with the perfect-unmixer as encoder when it comes to 1) ID d-predicting performance
    2) post-sparse-in-causal-mechanism-domain shift x-shot performance
    '''

    DATA_SETTING = 'MFP'

    def __init__(self, c_hid, DataClass, num_causal_vars=None, only_parents=False, fixed_encoder=None, coral=False, anti_align=False, no_intvs=False, cma_true_parentage=False, fixed_logstd=False, sample_mean=False, use_nonzero_mask=False, encourage_cfd=False,
        gumbel_temperature=1.0, standalone=True, **kwargs):
        super().__init__(DataClass, num_causal_vars)
        self.num_latents = self.num_fine_vars
        self.save_hyperparameters()
        assert not (fixed_encoder and anti_align)
        # self.num_coarse_vars = len(set(DataClass.FINE2COARSE.values())) MOVED TO PARENT
        # self.num_coarse_intv_vars = len(DataClass.INTERVENED_FACTORS) if not no_intvs else len(DataClass.FACTORS_FOR_NO_INTV_CASE)
        # self.num_causal_vars = num_causal_vars # size of intervention vector that is given in a batch. It decides how many factors / param_groups we'll try to disentangle
        # self.num_groups = self.num_causal_vars + 1 # +1 for information never affected by intervention, that is hence assumed undisentangleable, called the 'trash' group
        # self.num_fine_vars = len(DataClass.FINE2COARSE)
        # self.num_latents = self.num_fine_vars
        # self.fine_idx2coarse_idx = {DataClass.FACTORS.index(f): DataClass.COARSE_FACTORS.index(c) for f,c in DataClass.FINE2COARSE.items()}
        # if self.num_causal_vars == self.num_coarse_vars:
        #     self.fine_idx2group_idx = self.fine_idx2coarse_idx
        # elif self.num_causal_vars == self.num_fine_vars:
        #     self.fine_idx2group_idx = {i: i for i in range(self.num_fine_vars)}
        # else:
        #     raise NotImplementedError
        # self.short_factors = DataClass.SHORT_FACTORS MOVED TO PARENT
        self.register_buffer('mx_matrix', torch.tensor(DataClass.MIXING_MATRIX))
        # self.ux_matrix = self.mx_matrix.inverse()
        self.register_buffer('ux_matrix', self.mx_matrix.inverse())
        from models.shared.flow_layers import OrthogonalFlow
        self.fixed_encoder = fixed_encoder
        self.only_parents = only_parents
        self.gumbel_temperature = gumbel_temperature
        self.DA_loss_factor_keys = ['e_sum', 'd_sum', 'c_mean'] + [f'e_{i}' for i in range(self.num_latents)] + [f'c_{f}' for f in DataClass.SHORT_FACTORS]
        self.var_info = {k: v for (k, v) in self.hparams.DataClass.MY_VAR_INFO.items() if
                         k in self.hparams.DataClass.FACTORS}
        self.validation_step_outputs = {}
        # region encoder
        self.encoder = OrthogonalFlow(self.num_fine_vars, LU_decomposed=False) # Not doing LU decomposition is slower, but easier, and for now speed is not that important. This is identical to a biasless linear layer, but has the ldj-calculation built in.
        if self.fixed_encoder == 'perfect_ux':
            self.encoder.weight.data = self.ux_matrix.data
        elif self.fixed_encoder == 'identity':
            self.encoder.weight.data = torch.eye(self.num_fine_vars)
        #endregion

        # region context layer
        if only_parents:
            raise NotImplementedError("Not implemented cuz not sure if doing only-parents ID and OOD for fixed_encoder is comparable to all-parents non-fixed_encoder")
        else:
             self.context_layer = MultivarLinear(input_dims=self.num_latents, output_dims=c_hid, extra_dims=[self.num_latents])
        # endregion

        # region target layer
        self.target_layer = MultivarLinear(
            input_dims=self.num_causal_vars,
            # input_dims=self.num_groups,
            output_dims=c_hid,
            extra_dims=[self.num_latents]
        )
        # endregion

        # region net
        self.net = nn.Sequential(
            nn.SiLU(),
            MultivarLinear(input_dims=c_hid, output_dims=c_hid, extra_dims=[self.num_latents]),
            nn.SiLU(),
            MultivarLinear(input_dims=c_hid, output_dims=2, extra_dims=[self.num_latents])
        )
        # endregion

        # region Load causal encoder for correlation metrics logging
        if 'causal_encoder_checkpoint' in self.hparams:
            self.causal_encoder, self.causal_encoder_true_epoch = get_causal_encoder(self.hparams.causal_encoder_checkpoint)
            self.register_buffer('last_target_assignment', torch.zeros(self.num_latents, self.num_causal_vars))
        #endregion

        # region anti-aligner
        if anti_align:
            self.e_to_c_target_assignment = nn.Parameter(torch.zeros(self.num_fine_vars, self.num_fine_vars))
            self.e_to_c_predictor = CausalEncoder(c_hid=128,
                                                  lr=4e-3,
                                                  causal_var_info=OrderedDict(
                                                      {k: v for (k, v) in DataClass.MY_VAR_INFO.items() if
                                                       k in DataClass.FACTORS}),
                                                  single_linear=True,
                                                  c_in=self.num_fine_vars*2,
                                                  warmup=0)
            self.e_to_c_optimizer = self.e_to_c_predictor.get_main_optimizer()
        #endregion
        self.cma_true_parentage = cma_true_parentage
        self.use_nonzero_mask = use_nonzero_mask
        if self.cma_true_parentage:
            assert self.fixed_encoder == 'perfect_ux'
            # Mask: rows are factors, columns are parents
            mask = torch.tensor([[1 if p in DataClass.ID_PARENTAGE[f] else 0 for p in DataClass.FACTORS] for f in DataClass.FACTORS])
            self.context_layer.input_mask = mask
            self.context_layer.use_nonzero_mask = self.use_nonzero_mask
        self.fixed_logstd = fixed_logstd
        self.sample_mean = sample_mean
        if self.fixed_encoder == 'perfect_ux':
            # if perfect_ux, num_causal_vars should equal EITHER num_fine_vars OR num_coarse_vars, and num_latents should equal num_fine_vars
            assert self.num_causal_vars in [self.num_fine_vars, self.num_coarse_vars]
            assert self.num_latents == self.num_fine_vars
            if self.num_causal_vars == self.num_fine_vars:
                psi = torch.cat([torch.eye(self.num_fine_vars), torch.zeros(self.num_fine_vars, 1)], dim=1)
            elif self.num_causal_vars == self.num_coarse_vars:
                psi = torch.zeros(self.num_latents, self.num_groups)
                # use self.fine_idx2coarse_idx to fill in psi. This will leave the trash group as all zeros.
                for fine_idx, coarse_idx in self.fine_idx2coarse_idx.items():
                    psi[fine_idx, coarse_idx] = 1
            # psi_params such that softmax(psi_params) = psi
            self.psi_params = nn.Parameter(torch.where(psi == 0, torch.tensor(-1000.).to(psi.device), torch.tensor(0.).to(psi.device)))
        else:
            self.psi_params = nn.Parameter(torch.zeros(self.num_latents, self.num_groups))
        self.register_buffer('eye_num_groups', torch.eye(self.num_groups))
        self.register_buffer('eye_causal_vars_num_groups', torch.eye(self.num_groups)[:, :self.num_causal_vars])


    @property
    def psi(self):
        # hard argmax
        return F.one_hot(torch.argmax(self.psi_params, dim=-1), num_classes=self.num_groups).float()

    def get_target_assignment(self, hard=False):
        # Returns psi, either 'hard' (one-hot, e.g. for triplet eval) or 'soft' (probabilities, e.g. for debug)
        if not hard:
            return torch.softmax(self.psi_params, dim=-1)
        else:
            return F.one_hot(torch.argmax(self.psi_params, dim=-1), num_classes=self.num_groups)


    def get_psi_params(self):
        return self.psi_params

    def reset_matching_parent_mask(self, idx):
        self.context_layer.input_mask[idx, :] = 1

    VAL_TRACK_METRIC = 'val_nll_d_sum'

    # def get_target_assignment(self, hard=False):
    #     # Returns psi, either 'hard' (one-hot, e.g. for triplet eval) or 'soft' (probabilities, e.g. for debug)
    #     if not hard:
    #         return torch.softmax(self.target_params, dim=-1)
    #     else:
    #         return F.one_hot(torch.argmax(self.target_params, dim=-1), num_classes=self.target_params.shape[-1]

    def predict_future_latent_params(self, et, It1):
        '''

        :param et:
        :param It1: [B, num_causal_vars]
        :return:
        '''
        citris_tricks = hasattr(self, 'citris_tricks') and self.citris_tricks
        # region context_feats
        # Transform et into a feature vector.
        context_feats = self.context_layer(et)
        if citris_tricks:
            raise NotImplementedError("TODO for CI-MFP, masking context layer in different hypothesis based on if the intervention for that hypothesis is 1 (and so never for the trash hypothesis)")
            context_feats = context_feats[:, None, ...].expand(-1, self.num_groups, -1, -1).reshape(-1, context_feats.shape[-2], context_feats.shape[-1])
        # endregion
        # region target_feats
        # psi_samples = F.gumbel_softmax(self.psi_params[None].expand(It1.shape[0], -1, -1),
        #                                   tau=self.gumbel_temperature, hard=True) # [B, num_latents, num_groups]

        if not citris_tricks:
            target_feats = self.target_layer(It1)
        else:
            raise NotImplementedError("TODO implement citris_tricks fully, eg psi-based selection among hypotheses")
            It1_per_hyp = It1[:, None, :] * self.eye_causal_vars_num_groups[None]  # [B, num_groups, num_causal_vars]
            It1_per_hyp = It1_per_hyp - (1 - self.eye_causal_vars_num_groups[None]) # This is a bit different than RVP, where It1_per_hyp has dimension [num_groups, num_groups]. But the trash-group there just always sees [-1, -1, ..., 0], while here it sees [-1, -1, ..., -1], which isn't fundamentally different imo. And this allows me to remain backwards compatible.
            # stack first two dims to get [B*num_groups, num_groups]
            It1_per_hyp = It1_per_hyp.reshape(-1, self.num_causal_vars)
            # #unstack
            # It1_and_trash_per_hyp = It1_and_trash_per_hyp.reshape(It1.shape[0], self.num_groups, self.num_groups)
            target_feats = self.target_layer(It1_per_hyp)
        # endregion

        # Sum all features and use as input to feature network (division by 2 for normalization)
        feats = (target_feats + context_feats) / 2.0
        et1_pred_params = self.net(
            feats)
        # unstack
        if citris_tricks:
            et1_pred_params = et1_pred_params.reshape(It1.shape[0], self.num_groups, et1_pred_params.shape[-2], et1_pred_params.shape[-1])
        return et1_pred_params

    def encode(self, d_or_c, from_c=False):
        if from_c:
            d = d_or_c @ self.mx_matrix
        else:
            d = d_or_c
        if self.fixed_encoder is not None:
            with torch.no_grad():
                e, ldj = self.encoder(d)
        else:
            e, ldj = self.encoder(d)
        return e, ldj

    def decode(self, e):
        if self.fixed_encoder is not None:
            with torch.no_grad():
                d = self.encoder.reverse(e)
        else:
            d = self.encoder.reverse(e)
        return d

    def get_losses(self, batch, args = None, pretrain_phase=False):

        coral=False
        batch, coral, id_batch_size, ood_batch_size = maybe_id_and_ood_unpack(batch, coral)
        if coral:
            raise NotImplementedError()
        dt, dt1, It1 = self.unpack_and_mix_batch(batch)
        (et,_), (et1,ldjt1) = self.encode(dt), self.encode(dt1)
        et1_pred_params = self.predict_future_latent_params(et, It1)
        et1_pred_mean, et1_pred_logstd = et1_pred_params[..., 0], et1_pred_params[..., 1]
        if self.fixed_logstd:
            et1_pred_logstd = torch.zeros_like(et1_pred_logstd)


        nll_per_et1 = -gaussian_log_prob(mean=et1_pred_mean, log_std=et1_pred_logstd, samples=et1,
                                         # std_minimum=self.hparams.std_min
                                         std_minimum=(self.hparams.std_min if 'std_min' in self.hparams else 0) # For compatibility with old checkpoints in which std_min was not used
                                         )
        nll_et1_sum = nll_per_et1.sum(-1)
        nll_dt1_sum = nll_et1_sum + ldjt1
        loss_dict = {
            'nll_per_e': nll_per_et1.mean(dim=0), # mean over batch dim
            'nll_d_sum': nll_dt1_sum.mean(dim=0), # mean over batch dim
            'nll_e_sum': nll_et1_sum.mean(dim=0), # mean over batch dim
        }
        loss_dict['mse_per_et1'] = ((et1_pred_mean - et1)**2).mean(0)
        loss_dict['mse_et1_mean'] = loss_dict['mse_per_et1'].mean()
        loss_dict['logstd_per_et1'] = et1_pred_logstd.mean(0)
        loss_dict['minlogstd_per_et1'] = et1_pred_logstd.min(0)[0]
        loss_dict['logstd_et1_mean'] = loss_dict['logstd_per_et1'].mean()
        loss_dict['minlogstd_et1'] = loss_dict['minlogstd_per_et1'].min()
        with torch.no_grad():
            # dt1_pred_mean = self.decode(et1_pred_mean)
            et1_noise = torch.randn_like(et1_pred_logstd)
            et1_pred_sample = et1_pred_mean
            sample_mean = self.sample_mean or (args is not None and args.sample_mean) # This is a bit ugly: the self. way is during id training, the args. way is during domain adaptation
            if not sample_mean:
                et1_pred_sample += et1_noise * torch.exp(et1_pred_logstd)
            dt1_pred_sample = self.decode(et1_pred_sample)
            loss_dict['mse_per_dt1'] = ((dt1_pred_sample - dt1)**2).mean(0)
            loss_dict['mse_dt1_mean'] = loss_dict['mse_per_dt1'].mean()

            ct1_pred_sample = dt1_pred_sample @ self.ux_matrix
            # # Cap ct1_pred_sample between -2.0 on the one hand, +2*pi on the other hand
            # capped_ct1_pred_sample = torch.where(ct1_pred_sample < -2.0,
            #                                      torch.ones_like(ct1_pred_sample)*-2.0,
            #                                      torch.where(ct1_pred_sample > 2*np.pi,
            #                                                  torch.ones_like(ct1_pred_sample)*2*np.pi,
            #                                                  ct1_pred_sample))
            ct1_truth = dt1 @ self.ux_matrix
            _, loss_dict['normed_dist_per_c']  = self.c_dist(ct1_pred_sample, ct1_truth)
            loss_dict['normed_dist_c_mean'] = loss_dict['normed_dist_per_c'].mean()

        loss_dict['fsl_train_loss'] = loss_dict['nll_d_sum'].clone() # fsl stands for few-shot learning
        loss_dict['pretrain_loss'] = loss_dict['nll_d_sum'].clone()
        # loss_dict['fsl_train_loss'] = loss_dict['nll_d_sum']
        # loss_dict['pretrain_loss'] = loss_dict['nll_d_sum']
        if coral:
            #concat prior mean and logstd in z-dim
            coral_values = et1_pred_params.flatten(-2, -1) # (Bi+Bo)_C_2 -> (Bi+Bo)_C*2
            id_values = coral_values[:id_batch_size] # (Bi+Bo)_C*2 -> Bi_C*2
            ood_values = coral_values[id_batch_size:] # (Bi+Bo)_C*2 -> Bo_C*2
            l_coral = coral_loss(id_values, ood_values)
            loss_dict['l_coral'] = l_coral
            loss_dict['fsl_train_loss'] += l_coral*self.hparams.beta_coral
        if self.hparams.anti_align:
            ctt1 = torch.cat((batch[-1][:,0],batch[-1][:,1]),0) # corralled target
            ett1 = torch.cat((et, et1))
            ett1_exp, ctt1_exp = expand_per_group(inps=ett1,target_assignment=torch.eye(self.num_fine_vars).to(ett1.device),latents=ctt1, flatten_inp=False)

            # Get avg loss, which e_to_c_predictor will minimize, and which other parameters will not be affected by
            v_dict = self.e_to_c_predictor(ett1_exp.detach())
            loss_per_c, _, _, _ = self.e_to_c_predictor.calculate_loss_distance(v_dict, ctt1_exp,also_unaveraged_loss=True)
            avg_loss = sum([loss_per_c[key] for key in loss_per_c])

            # Get min loss, which other parameters will maximize, and which e_to_c_predictor will not be affected by
            self.e_to_c_predictor.requires_grad_(False)
            v_dict = self.e_to_c_predictor(ett1_exp)
            self.e_to_c_predictor.requires_grad_(True)
            _, _, _, unaveraged_loss_per_c = self.e_to_c_predictor.calculate_loss_distance(v_dict, ctt1_exp,also_unaveraged_loss=True)
            min_loss_per_c = {factor: torch.min(val,-1)[0].mean() for factor, val in unaveraged_loss_per_c.items()}
            avg_min_loss = sum([min_loss_per_c[key] for key in min_loss_per_c])
            loss_dict['l_anti_align'] = avg_loss - avg_min_loss
            loss_dict['l_anti_align_avg'] = avg_loss
            loss_dict['l_anti_align_min'] = avg_min_loss
            loss_dict['pretrain_loss'] += loss_dict['l_anti_align']*self.hparams.beta_anti_align

        return loss_dict

    def c_dist(self, c_pred, c_truth):
        c_dists = torch.zeros(self.num_fine_vars)
        norm_c_dists = torch.zeros_like(c_dists)
        for i, info in enumerate(self.var_info.values()):
            if info.startswith('continuous'): # mse
                c_dists[i] = ((c_pred[:,i] - c_truth[:,i])**2).mean()
                scale = info.split('_')[-1] * 2
                norm_c_dists[i] = c_dists[i] / float(scale)
            elif info.startswith('angle'):
                c_dists[i] = (((c_pred[:, i] % (2*np.pi)) - (c_truth[:, i] % (2*np.pi)))**2).mean()
                norm_c_dists[i] = c_dists[i] / (2*np.pi)
            elif info.startswith('categ'):
                c_dists[i] = (c_pred[:,i].round() != c_truth[:,i].round()).float().mean()
                norm_c_dists[i] = c_dists[i]
            else:
                raise ValueError(f'Unknown var_info {info}')
        return c_dists, norm_c_dists


    # def get_FSL_train_loss(self, batch):
    #     loss_dict = self.get_losses(batch)
    #     loss = loss_dict['fsl_train_loss']
    #     return loss


    def prep_dict_for_log(self, dataloader_idx, loss_dict, split):
        maybe_cma_prefix = 'cma_' if dataloader_idx > 0 else ''
        dict_to_log = {f'{maybe_cma_prefix}{split}_{key}': val for key, val in loss_dict.items() if
                       not '_per_' in key} | \
                      {f'{maybe_cma_prefix}{split}_' + key.replace('_per', '') + f'_{i}': subval for key, val in
                       loss_dict.items() if '_per_' in key for i, subval in enumerate(val)}
        return dict_to_log

    def unpack_and_mix_batch(self, batch):
        # _, It1, ctt1 = batch
        It1, ctt1 = batch['targets'], batch['lat']
        It1 = It1.squeeze(1).float()
        ct, ct1 = ctt1[:, 0], ctt1[:, 1]
        dt, dt1 = ct @ self.mx_matrix, ct1 @ self.mx_matrix
        return dt, dt1, It1

    def stage_step(self, batch, stage, dataloader_idx=0):
        # iid_pairs = (dataloader_idx == 1)
        loss_dict = self.get_losses(batch) #, iid_pairs=iid_pairs)
        dict_to_log = self.prep_dict_for_log(dataloader_idx, loss_dict, stage)
        self.log_dict(dict_to_log, add_dataloader_idx=False)
        if stage == 'val':
            for key, val in dict_to_log.items():
                if key in self.validation_step_outputs:
                    self.validation_step_outputs[key] += [val]
                else:
                    self.validation_step_outputs[key] = [val]
            maybe_cma_prefix = 'cma_' if dataloader_idx > 0 else ''
            key = f'{maybe_cma_prefix}num_samples'
            if key in self.validation_step_outputs:
                self.validation_step_outputs[key] += [len(batch['targets'])]
            else:
                self.validation_step_outputs[key] = [len(batch['targets'])]
        return loss_dict['pretrain_loss']

    def training_step(self, batch, batch_idx):
        return self.stage_step(batch, 'train')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.stage_step(batch, 'test', dataloader_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.stage_step(batch, 'val', dataloader_idx)

    def on_validation_epoch_end(self):
        for key, val_list in self.validation_step_outputs.items():
            # self.log(key, torch.mean(torch.stack(val_list)))
            if 'num_samples' in key:
                continue
            counts = self.validation_step_outputs['cma_num_samples'] if key.startswith('cma_') else self.validation_step_outputs['num_samples']
            counts = torch.tensor(counts).to(val_list[0].device)
            agg_value = torch.sum(torch.stack(val_list) * counts) / torch.sum(counts)
            self.log(key, agg_value)
        self.validation_step_outputs = {}


    @classmethod
    def get_specific_cormet_args(cls, *args, **kwargs):
        return {
            "mfp": True,
        }

    @classmethod
    def get_specific_callbacks(cls, *args, **kwargs):
        return []

    # DEPRECATED: refactored into superclass
    # @staticmethod
    # def get_callbacks(**kwargs):
    #     callbacks = []
    #     # Create learning rate callback
    #     callbacks.append(LearningRateMonitor('step'))
    #
    #     if not kwargs['no_alignment_measuring']:
    #         # Correlation Metrics Logging Callback
    #         cormet_args = {
    #             "dataset":  kwargs['correlation_dataset'],
    #             "cluster": kwargs['cluster'],
    #             "test_dataset": kwargs['correlation_test_dataset'],
    #             "mfp": True
    #         }
    #         if kwargs['mini']:
    #             cormet_args['num_train_epochs'] = 5
    #         callbacks.append(BaselineCorrelationMetricsLogCallback(**cormet_args))
    #
    #     return callbacks

    def configure_optimizers(self, max_iters=None):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters if max_iters is None else max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def get_sparse_update_params(self):
        # Excludes parameters of  self.causal_encoder. Return as iterator
        return ((n,p) for n, p in self.named_parameters() if not n.startswith('causal_encoder'))


    def is_shared(self, param_name):
        return False

    def get_unfreeze_idx(self, unfreeze_setting):
        if unfreeze_setting == 'uf_all':
            unfreeze_idx = 'uf_all'
        else:
            ext = unfreeze_setting.split('uf_')[1]
            if ext.isdigit():
                unfreeze_idx = int(ext)
            else:
                unfreeze_idx = self.short_factors.index(ext)
        return unfreeze_idx

    def get_grouped_flp_params(self, clone=True):
        # params = {'shared': []} | {i: [] for i in range(self.num_fine_vars)} # no shared params atm
        params = {i: [] for i in range(self.num_fine_vars)}
        for name, param in self.named_parameters():
            if any([substr in name for substr in ['encoder','e_to_c', 'psi']]):
                continue
            elif any([substr in name for substr in ['context_layer', 'target_layer', 'net']]):
                for i, param_slice in enumerate(param):
                    params[i].append(param_slice.clone() if clone else param_slice)
            else:
                raise ValueError(f'Unknown parameter group for {name}')
        return params

    @property
    def prior_parts(self):
        return [self.context_layer, self.target_layer, self.net]

    def fsl_calibration_metric(self, args):
        return 'mse_per_et1'

    @property
    def psi_probs(self):
        return torch.softmax(self.psi_params, dim=-1)

