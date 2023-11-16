"""
Summarizing all normalizing flow layers
"""
import socket
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from collections import defaultdict, OrderedDict
import scipy.linalg
import pytorch_lightning as pl
import sys
import glob

from models.shared.callbacks import expand_per_group, CorrelationMetricsLogCallback, \
    BaselineCorrelationMetricsLogCallback, LearningRateMonitor, ImageLogCallback
import os

from models.shared.causal_encoder import get_causal_encoder, grad_reverse
from models.shared.triplet_evaluator import TripletEvaluator, flow_based_triplet_pred, unpack_triplet_batch

sys.path.append('../../')
from models.shared import AutoregLinear, CosineWarmupScheduler, get_act_fn, CausalEncoder
from util import tn

# class AutoregNormalizingFlow(nn.Module):
class AutoregNormalizingFlow(TripletEvaluator):
    """ 
    Base class for the autoregressive normalizing flow 
    We use a combination of affine autoregressive coupling layers,
    activation normalization, and invertible 1x1 convolutions / 
    orthogonal invertible transformations.
    """

    def __init__(self, num_vars=None, num_flows=None, act_fn=None, hidden_per_var=16, zero_init=False, use_scaling=True, use_1x1_convs=True, init_std_factor=0.2,
                  lr=1e-3, warmup=500, max_iters=100000, DataClass=None, noise_level=-1.0, angle_reg_weight=0.1, lambda_grad_reverse=0.0, standalone=False,
                 # zero_nonmatch=False, straightthru_nonmatch=False,
                 nonmatch_stragey='gradreverse', triplet_train=False, z2c_triplet_ratio=.5,nd_as_loss=False, lambda_reg=0.0,
                 **kwargs):
        """

        :param num_vars: number of z-dimensions
        """
        super().__init__()
        self.save_hyperparameters() # for some reason this saves standalone=True even when it is False.
        # Backwards compatibility
        if act_fn is None:
            assert kwargs['flow_act_fn'] is not None
            act_fn = get_act_fn(kwargs['flow_act_fn'])
        if num_vars is None:
            assert kwargs['num_latents'] is not None
            num_vars = kwargs['num_latents']
        self.num_vars = num_vars

        if self.hparams.noise_level < 0.0:
            self.hparams.noise_level = self.get_autoencoder_noise_level()

        self.flows = nn.ModuleList([])
        transform_layer = lambda num_vars: AffineFlow(num_vars, use_scaling=use_scaling)
        for i in range(num_flows):
            self.flows.append(ActNormFlow(num_vars))
            if i > 0:
                if use_1x1_convs:
                    self.flows.append(OrthogonalFlow(num_vars))
                else:
                    self.flows.append(ReverseSeqFlow())
            self.flows.append(AutoregressiveFlow(num_vars, 
                                                 hidden_per_var=hidden_per_var, 
                                                 act_fn=act_fn, 
                                                 init_std_factor=(0 if zero_init else init_std_factor),
                                                 transform_layer=transform_layer))

        if standalone:

            from models.ae import Autoencoder
            self.z_to_c_predictor = CausalEncoder(c_hid=128,
                                                  lr=4e-3,
                                                  causal_var_info=OrderedDict(
                                                      {k: v for (k, v) in DataClass.MY_VAR_INFO.items() if
                                                       k in DataClass.FACTORS}),
                                                  single_linear=True,
                                                  c_in=num_vars * 2,
                                                  warmup=0,
                                                  angle_reg_weight=angle_reg_weight,)
            self.psi_params = nn.Parameter(torch.zeros(num_vars, len(DataClass.FACTORS)))

            if 'causal_encoder_checkpoint' in self.hparams:
                self.causal_encoder, self.causal_encoder_true_epoch = get_causal_encoder(self.hparams.causal_encoder_checkpoint)
            self.register_buffer('last_target_assignment', torch.zeros(num_vars, len(DataClass.FACTORS)))
            self.autoencoder = Autoencoder.load_from_checkpoint(self.hparams.autoencoder_checkpoint)
            for p in self.autoencoder.parameters():
                p.requires_grad_(False)

    def get_target_assignment(self, hard=False, grad_psi=False):
        if hard:
            if not grad_psi:
                return F.one_hot(torch.argmax(self.psi_params, dim=-1), num_classes=self.psi_params.shape[-1])
            else:
                # Use gumbel softmax
                return F.gumbel_softmax(self.psi_params, tau=1.0, hard=True)
        else:
            return F.softmax(self.psi_params, dim=-1)

    def get_autoencoder_noise_level(self):
        ae_ckpt = self.hparams.autoencoder_checkpoint # eg a/b/epoch=0.ckpt
        # ae_dir: a/b
        ae_dir = os.path.dirname(ae_ckpt)
        # check if file called noise_level__( the noise level) exists in ae_dir.
        # if it does, then return the noise level
        # if it doesn't, then load the ae model, retrieve the noise level, and save it to the file
        # use glob
        if len(glob.glob(os.path.join(ae_dir, 'noise_level__*'))) > 0:
            noise_level = float(glob.glob(os.path.join(ae_dir, 'noise_level__*'))[0].split('__')[-1])
        else:
            from models.ae import Autoencoder
            ae_model = Autoencoder.load_from_checkpoint(ae_ckpt)
            noise_level = ae_model.hparams.noise_level
            # create empty file called noise_level__(the noise level) in ae_dir
            with open(os.path.join(ae_dir, f'noise_level__{noise_level}'), 'w') as _:
                pass
        return noise_level



    def forward(self, x):
        ldj = x.new_zeros(x.shape[0],) # ldj stands for log det jacobian
        for flow in self.flows:
            x, ldj = flow(x, ldj)
        return x, ldj

    def reverse(self, x):
        for flow in reversed(self.flows):
            x = flow.reverse(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        if self.hparams.triplet_train:
            return self.z2c_and_triplet_loss_step(batch, batch_idx, 'train')
        else:
            return self.z2c_loss_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, stage='val', dataloader_idx=dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, stage='test', dataloader_idx=dataloader_idx)

    def eval_step(self, batch, batch_idx, stage, dataloader_idx=0):
        if dataloader_idx == 0:
            return self.z2c_loss_step(batch, batch_idx, stage, dataloader_idx=dataloader_idx)
        elif dataloader_idx == 1:
            return self.triplet_eval_step(batch, batch_idx, stage, dataloader_idx=dataloader_idx)

    def z2c_and_triplet_loss_step(self, batch, batch_idx, stage):
        imgs, labels, latents, obj_indices, source = unpack_triplet_batch(batch)
        y = imgs.reshape(-1, *imgs.shape[2:])
        z, ldj = self(y)
        # ldj_reg_loss = ldj.abs().mean()
        z_reg_loss = z.pow(2).mean()
        self.mylog(f'{stage}_ldj_mean', ldj.mean())
        self.mylog(f'{stage}_ldj_abs_mean', ldj.abs().mean())
        self.mylog(f'{stage}_z_reg_loss', z_reg_loss)
        # triplet part
        z_first_two = z.unflatten(0, imgs.shape[:2])[:,:2]
        triplet_rec = self.triplet_prediction(x_maybe_encs=None, source=source, grad_psi=True, z=z_first_two)
        self.causal_encoder.eval()
        # pred_dict = self.causal_encoder.forward(triplet_rec)
        # gt_vec = latents[:, -1]
        # losses = OrderedDict()
        # dists = OrderedDict()
        # norm_dists = OrderedDict()
        # nograd_dists = True
        # keep_sign = False
        # for i, var_key in enumerate(pred_dict):
        #     var_info = self.causal_encoder.hparams.causal_var_info[var_key]
        #     gt_val = gt_vec[..., i]
        #     if var_info.startswith('continuous'):
        #         # MSE loss
        #         losses[var_key] = F.mse_loss(pred_dict[var_key].squeeze(dim=-1),
        #                                      gt_val, reduction='none')
        #         with torch.no_grad() if nograd_dists else nullcontext():
        #             dists[var_key] = (pred_dict[var_key].squeeze(dim=-1) - gt_val)
        #             if not keep_sign:
        #                 dists[var_key] = dists[var_key].abs()
        #             norm_dists[var_key] = dists[var_key] / float(var_info.split('_')[-1])
        #     elif var_info.startswith('angle'):
        #         # Cosine similarity loss
        #         vec = torch.stack([torch.sin(gt_val), torch.cos(gt_val)], dim=-1)
        #         cos_sim = F.cosine_similarity(pred_dict[var_key], vec, dim=-1)
        #         losses[var_key] = 1 - cos_sim
        #         if self.causal_encoder.training:
        #             norm = pred_dict[var_key].norm(dim=-1, p=2.0)
        #             losses[var_key + '_reg'] = self.causal_encoder.hparams.angle_reg_weight * (2 - norm) ** 2
        #         with torch.no_grad() if nograd_dists else nullcontext():
        #             dists[var_key] = torch.where(cos_sim > (1 - 1e-7), torch.zeros_like(cos_sim),
        #                                          torch.acos(cos_sim.clamp_(min=-1 + 1e-7, max=1 - 1e-7)))
        #             dists[var_key] = dists[var_key] / np.pi * 180.0  # rad to degrees
        #             norm_dists[var_key] = dists[var_key] / 180.0
        #     elif var_info.startswith('categ'):
        #         # Cross entropy loss
        #         gt_val = gt_val.long()
        #         pred = pred_dict[var_key]
        #         if len(pred.shape) > 2:
        #             pred = pred.flatten(0, -2)
        #             gt_val = gt_val.flatten(0, -1)
        #         losses[var_key] = F.cross_entropy(pred, gt_val, reduction='none')
        #         # if len(pred_dict[var_key]) > 2: # Nathan: this fails if the batch size is 1
        #         if len(pred_dict[var_key].shape) > 2:
        #             losses[var_key] = losses[var_key].reshape(pred_dict[var_key].shape[:-1])
        #             gt_val = gt_val.reshape(pred_dict[var_key].shape[:-1])
        #         with torch.no_grad() if nograd_dists else nullcontext():
        #             dists[var_key] = (gt_val != pred_dict[var_key].argmax(dim=-1)).float()
        #             norm_dists[var_key] = dists[var_key]
        #     else:
        #         assert False, f'Do not know how to handle key \"{var_key}\" in calculating distances and losses.'
        # # if also_unaveraged_loss:
        # #     unaveraged_losses = {k: v.clone() for k, v in losses.items()}
        # #     extra_return = [unaveraged_losses]
        # # else:
        # #     extra_return = []
        # for var_key in losses:
        #     losses[var_key] = losses[var_key].mean()
        # # return [losses, dists, norm_dists] + extra_return
        triplet_losses, _, norm_dists, _ = self.causal_encoder.get_distances(triplet_rec, latents[:, -1], return_norm_dists=True, return_v_dict=True, avoid_storing=True)
        triplet_loss = torch.stack(list(triplet_losses.values())).mean()
        # triplet_norm_dist = sum([norm_dists[key].mean() for key in losses])
        triplet_norm_dist = torch.stack(list(norm_dists.values())).mean()
        self.mylog(f'{stage}_triplet_cnd', triplet_norm_dist)
        self.mylog(f'{stage}_triplet_loss', triplet_loss)

        if self.hparams.nd_as_loss:
            triplet_loss = triplet_norm_dist

        # z2c part
        # y_sample = imgs + torch.randn_like(imgs) * self.hparams.noise_level
        # stack first two dims (batch and triplet) together into batch dim
        # y_sample = y_sample.reshape(-1, *y_sample.shape[2:])
        c = latents.reshape(-1, *latents.shape[2:])
        # z, _ = self(y_sample)
        z2c_loss_dict = self.cheat_cfd_loss(z, c)
        self.log_z2c_loss_dict(z2c_loss_dict, stage)

        # total loss
        ratio = self.hparams.z2c_triplet_ratio
        loss = ratio * z2c_loss_dict['z2c_loss'] + (1 - ratio) * triplet_loss + self.hparams.lambda_reg_z * z_reg_loss
        self.mylog(f'{stage}_loss', loss)
        return loss

    # def on_train_epoch_end(self) -> None:
        # if True: #socket.gethostname() == "theoden":
        #     torch.cuda.empty_cache()  # A very hacky way to avoid CUDA out of memory error #on theoden



    def log(self, *args, add_dataloader_idx=False, **kwargs):
        # Only difference with default log is that add_dataloader_idx is False by default
        super().log(*args, add_dataloader_idx=add_dataloader_idx, **kwargs)

    def z2c_loss_step(self, batch, batch_idx, stage, dataloader_idx=0):
        y, c = batch['img'], batch['lat']
        y_sample = y + torch.randn_like(y) * self.hparams.noise_level
        z, _ = self(y_sample)
        loss_dict = self.cheat_cfd_loss(z, c)
        self.log_z2c_loss_dict(loss_dict, stage)
        return loss_dict['z2c_loss']

    def log_z2c_loss_dict(self, loss_dict, stage):
        self.mylog(f'{stage}_z2c_loss', loss_dict['z2c_loss'])
        for k, v in loss_dict['loss_per_c'].items():
            self.mylog(f'per_cdim/{stage}_{k}_loss_per_c', v)
        self.mylog(f'{stage}_norm_dists_diag_mean', loss_dict['diag_norm_dist'])
        self.mylog(f'{stage}_norm_dists_offdiag_mean', loss_dict[
            'offdiag_norm_dist'])  # sum of off-diag elements / (batch_size * num_factors * (num_factors - 1)) aka / number of off-diag elements
        self.mylog(f'{stage}_diag_loss', loss_dict['diag_loss'])
        self.mylog(f'{stage}_offdiag_loss', loss_dict['off_diag_loss'])
        self.mylog(f'{stage}_reg_loss', loss_dict['reg_loss'])

    def triplet_eval_step(self, batch, batch_idx, stage, dataloader_idx=0):

        loss_learned_target_assignment = self.triplet_evaluation(batch, mode=stage, dataloader_idx=dataloader_idx,
                                                                 use_cheating_target_assignment=False)
        loss_cheating_target_assignment = self.triplet_evaluation(batch, mode=stage, dataloader_idx=dataloader_idx,
                                                                  use_cheating_target_assignment=True)
        # self.log(f'{stage}_triplet_cnd', loss_learned_target_assignment)
        self.mylog(f'{stage}_triplet_cnd', loss_learned_target_assignment)
        # self.log(f'{stage}_triplet_cnd_cheating', loss_cheating_target_assignment)
        self.mylog(f'{stage}_triplet_cnd_cheating', loss_cheating_target_assignment)

    def cheat_cfd_loss(self, z, c):
        target_assignment = self.get_target_assignment(hard=True, grad_psi=True)
        z_exp, c_exp = expand_per_group(z, target_assignment, c, flatten_inp=False, detach=False)

        v_dict = OrderedDict()
        for i, var_key in enumerate(self.z_to_c_predictor.hparams.causal_var_info):
            z_match_selector = torch.arange(z_exp.shape[-2])[None, :, None].expand(z_exp.shape[0], -1, z_exp.shape[-1]).to(z_exp.device) == i
            if self.hparams.nonmatch_strategy == 'zero':
                z_for_pred = torch.where(z_match_selector, z_exp, torch.zeros_like(z_exp))
            elif self.hparams.nonmatch_strategy == 'straightthru':
                z_for_pred = z_exp
            elif self.hparams.nonmatch_strategy == 'gradreverse':
                z_gradrev = torch.where(z_match_selector, z_exp, grad_reverse(z_exp))
                z_straightthru = z_exp
                # cat along batch dim
                batch_size = z_exp.shape[0]
                z_for_pred = torch.cat([z_gradrev, z_straightthru], dim=0)
            else:
                raise ValueError(f'Unknown nonmatch strategy {self.hparams.nonmatch_strategy}')

            z_for_pred_enc = self.z_to_c_predictor.encoder(z_for_pred)
            v_dict[var_key] = self.z_to_c_predictor.pred_layers[var_key](z_for_pred_enc)
        if self.hparams.nonmatch_strategy == 'gradreverse':
            c_exp = torch.cat([c_exp, c_exp], dim=0)
        loss_per_c, _, norm_dists, loss_per_c_per_z = self.z_to_c_predictor.calculate_loss_distance(v_dict, c_exp, also_unaveraged_loss=True)
        if any(['reg' in k for k in loss_per_c_per_z]):
            if self.hparams.nonmatch_strategy == 'gradreverse':
                # only take the second half of the batch, which is the straightthru part
                # reg_loss = torch.tensor([loss_per_c_per_z[k][batch_size:].mean() for k in loss_per_c_per_z if 'reg' in k]).mean()
                # Above doesn't allow gradient to pass
                reg_loss = torch.stack([loss_per_c_per_z[k][batch_size:].mean() for k in loss_per_c_per_z if 'reg' in k]).mean()
            else:
                reg_loss = torch.stack([loss_per_c_per_z[k].mean() for k in loss_per_c_per_z if 'reg' in k]).mean()
        else:
            reg_loss = 0.0
        non_reg_loss_per_c_per_z = {k: v for k, v in loss_per_c_per_z.items() if 'reg' not in k}
        unpack_batch = (self.hparams.nonmatch_strategy == 'gradreverse')
        diag_loss, off_diag_loss = get_on_off_diag_means_for_tensor_dict(non_reg_loss_per_c_per_z, unpack_batch)
        diag_norm_dist, only_offdiag_norm_dist = get_on_off_diag_means_for_tensor_dict(norm_dists, unpack_batch)
        # if self.hparams.zero_nonmatch or self.hparams.straightthru_nonmatch:
        if self.hparams.nonmatch_strategy in ['zero', 'straightthru']:
            loss = torch.stack(list(loss_per_c.values())).mean()
        else:
            loss = diag_loss + self.hparams.lambda_grad_reverse * off_diag_loss + reg_loss


        return {'z2c_loss': loss, 'loss_per_c': loss_per_c, 'norm_dists': norm_dists, "diag_loss": diag_loss, "off_diag_loss": off_diag_loss, "reg_loss": reg_loss, "diag_norm_dist": diag_norm_dist, "offdiag_norm_dist": only_offdiag_norm_dist}


    @classmethod
    def get_callbacks(cls, exmp_triplet_inputs=None, correlation_dataset=None, correlation_test_dataset=None, mini=False, skip_correnc_train=False,**kwargs):
        callbacks = []
        callbacks.append(LearningRateMonitor('step'))
        callbacks += cls.add_cormet_callbacks(correlation_dataset, correlation_test_dataset, mini, skip_correnc_train=skip_correnc_train)
        callbacks.append(
            ImageLogCallback(exmp_triplet_inputs=exmp_triplet_inputs, dataset=correlation_dataset, every_n_epochs=kwargs['img_callback_freq'], skip_frame_prediction=True))
        return callbacks


    @classmethod
    def add_cormet_callbacks(cls, correlation_dataset, correlation_test_dataset, mini, skip_correnc_train=False):
        cormet_callbacks = []
        cormet_args = {
            "dataset": correlation_dataset,
            "cluster": False,
            "test_dataset": correlation_test_dataset,
            "mfp": False,
        }
        if mini:
            cormet_args['num_train_epochs'] = 5
        if skip_correnc_train:
            cormet_args['num_train_epochs'] = 0
        cormet_callbacks.append(CorrelationMetricsLogCallback(**cormet_args))
        cormet_callbacks.append(BaselineCorrelationMetricsLogCallback(**cormet_args | {"ignore_learnt_psi": True}))
        return cormet_callbacks


    def encode(self, x_or_y, random=True):
        if x_or_y.shape[-1] != self.num_vars:
            # raise NotImplementedError("Letting flow encode images is not supported yet, not storing the associated AE atm.")
            x = x_or_y
            y = self.autoencoder.encoder(x)
        else:
            y = x_or_y
        # Map input to disentangled latents, e.g. for correlation metrics
        if random:
            y = y + torch.randn_like(y) * self.hparams.noise_level
        z, ldj = self(y)
        return z

    def triplet_prediction(self, x_maybe_encs, source, use_cheating_target_assignment=False, grad_psi=False, z=None):
        """ Generates the triplet prediction of input encoding pairs and causal mask
        """
        # extracted to avoid code duplication with train_enc_cheat.py
        return flow_based_triplet_pred(self.autoencoder, flow=self, num_latents=self.hparams.num_latents, source=source, target_assignment=self.get_maybe_cheating_target_assignment(use_cheating_target_assignment, grad_psi=grad_psi), x_encs=x_maybe_encs, z=z)

    def get_maybe_cheating_target_assignment(self, use_cheating_target_assignment, grad_psi=False):
        assert not (use_cheating_target_assignment and grad_psi), "Can't use both cheating target assignment and grad psi"
        if use_cheating_target_assignment:
            return self.last_target_assignment.to(self.psi_params.device)
        else:
            return self.get_target_assignment(hard=True, grad_psi=grad_psi)

class AffineFlow(nn.Module):
    """ Affine transformation """

    def __init__(self, num_vars, use_scaling=True, hard_limit=-1):
        super().__init__()
        self.num_vars = num_vars
        self.hard_limit = hard_limit
        self.use_scaling = use_scaling
        if self.use_scaling and self.hard_limit <= 0:
            self.scaling = nn.Parameter(torch.zeros(num_vars,))

    def get_num_outputs(self):
        return 2

    def _get_affine_params(self, out):
        if isinstance(out, (list, tuple)):
            t, s = out
        else:
            t, s = out.unflatten(-1, (-1, 2)).unbind(dim=-1)
        if self.use_scaling:
            if self.hard_limit > 0:
                s = s - torch.max(s - self.hard_limit, torch.zeros_like(s)).detach()
                s = s + torch.max(-self.hard_limit - s, torch.zeros_like(s)).detach()
            else:
                sc = torch.tanh(self.scaling.exp()[None] / 3.0) * 3.0
                s = torch.tanh(s / sc.clamp(min=1.0)) * sc
        else:
            s = s * 0.0
        return t, s

    def forward(self, x, out, ldj):
        t, s = self._get_affine_params(out)
        x = (x + t) * s.exp()
        ldj = ldj - s.sum(dim=1)
        return x, ldj

    def reverse(self, x, out):
        t, s = self._get_affine_params(out)
        x = x * (-s).exp() - t
        return x


class AutoregressiveFlow(nn.Module):
    """ Autoregressive flow with arbitrary invertible transformation """

    def __init__(self, num_vars, hidden_per_var=16, 
                       act_fn=lambda: nn.SiLU(),
                       init_std_factor=0.2,
                       transform_layer=AffineFlow):
        super().__init__()
        self.transformation = transform_layer(num_vars)
        self.net = nn.Sequential(
                AutoregLinear(num_vars, 1, hidden_per_var, diagonal=False),
                act_fn(),
                AutoregLinear(num_vars, hidden_per_var, hidden_per_var, diagonal=True),
                act_fn(),
                AutoregLinear(num_vars, hidden_per_var, self.transformation.get_num_outputs(), diagonal=True,
                              no_act_fn_init=True, 
                              init_std_factor=init_std_factor, 
                              init_bias_factor=0.0,
                              init_first_block_zeros=True)
            )

    def forward(self, x, ldj):
        out = self.net(x)
        x, ldj = self.transformation(x, out, ldj)
        return x, ldj

    def reverse(self, x):
        inp = x * 0.0
        for i in range(x.shape[1]):
            out = self.net(inp)
            x_new = self.transformation.reverse(x, out)

            # inp[:,i] = x_new[:,i]
            x_new_slice = torch.zeros_like(x_new)
            x_new_slice[:,i] = x_new[:,i]
            inp = inp + x_new_slice # Nathan: when backpropping through this (as I do for domain_adaptation), you can't edit inp itself.

        return x_new


class ActNormFlow(nn.Module):
    """ Activation normalization """

    def __init__(self, num_vars):
        super().__init__()
        self.num_vars = num_vars 
        self.data_init = False

        self.bias = nn.Parameter(torch.zeros(self.num_vars,))
        self.scales = nn.Parameter(torch.zeros(self.num_vars,))
        self.affine_flow = AffineFlow(self.num_vars, hard_limit=3.0)

    def get_num_outputs(self):
        return 2

    def forward(self, x, ldj):
        if self.training and not self.data_init:
            self.data_init_forward(x)
        x, ldj = self.affine_flow(x, [self.bias[None], self.scales[None]], ldj)
        return x, ldj

    def reverse(self, x):
        x = self.affine_flow.reverse(x, [self.bias[None], self.scales[None]])
        return x

    @torch.no_grad()
    def data_init_forward(self, input_data):
        if (self.bias != 0.0).any():
            self.data_init = True
            return 

        batch_size = input_data.shape[0]

        self.bias.data = -input_data.mean(dim=0)
        self.scales.data = -input_data.std(dim=0).log()
        self.data_init = True

        out, _ = self.forward(input_data, input_data.new_zeros(batch_size,))
        print(f"[INFO - ActNorm] New mean: {out.mean().item():4.2f}")
        print(f"[INFO - ActNorm] New variance {out.std(dim=0).mean().item():4.2f}")


class ReverseSeqFlow(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, ldj):
        return torch.flip(x, dims=(-1,)), ldj

    def reverse(self, x):
        return self.forward(x, None)[0]


class OrthogonalFlow(nn.Module):
    """ Invertible 1x1 convolution / orthogonal flow """

    def __init__(self, num_vars, LU_decomposed=True):
        super().__init__()
        self.num_vars = num_vars
        self.LU_decomposed = LU_decomposed

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(self.num_vars, self.num_vars)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)

        if not self.LU_decomposed:
            self.weight = nn.Parameter(torch.from_numpy(w_init), requires_grad=True)
        else: 
            # LU decomposition can slightly speed up the inverse
            np_p, np_l, np_u = scipy.linalg.lu(w_init) # Permutation, lower, upper
            np_s = np.diag(np_u) # Diagonal of upper
            np_sign_s = np.sign(np_s) # Sign of diagonal of upper
            np_log_s = np.log(np.abs(np_s)) # Log of absolute value of diagonal of upper
            np_u = np.triu(np_u, k=1) # Zeroes out the diagonal of upper
            l_mask = np.tril(np.ones(w_init.shape, dtype=np.float32), -1) # 1 for lower triangular excluding diagonal
            eye = np.eye(*w_init.shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)), requires_grad=True)
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)), requires_grad=True)
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)), requires_grad=True)
            self.register_buffer('l_mask', torch.Tensor(l_mask))
            self.register_buffer('eye', torch.Tensor(eye))

        self.eval_dict = defaultdict(lambda : self._get_default_inner_dict())

    def _get_default_inner_dict(self):
        return {"weight": None, "inv_weight": None, "sldj": None}

    def _get_weight(self, device_name, inverse=False):
        if self.training or self._is_eval_dict_empty(device_name):
            if not self.LU_decomposed:
                weight = self.weight
                sldj = torch.slogdet(weight)[1]
            else:
                l, log_s, u = self.l, self.log_s, self.u
                l = l * self.l_mask + self.eye
                u = u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(log_s))
                weight = torch.matmul(self.p, torch.matmul(l, u))
                sldj = log_s.sum()
        
        if not self.training:
            if self._is_eval_dict_empty(device_name):
                self.eval_dict[device_name]["weight"] = weight.detach()
                self.eval_dict[device_name]["sldj"] = sldj.detach()
                self.eval_dict[device_name]["inv_weight"] = torch.inverse(weight.double()).float().detach()
            else:
                weight, sldj = self.eval_dict[device_name]["weight"], self.eval_dict[device_name]["sldj"]
        elif not self._is_eval_dict_empty(device_name):
            self._empty_eval_dict(device_name)
        
        if inverse:
            if self.training:
                weight = torch.inverse(weight.double()).float()
            else:
                weight = self.eval_dict[device_name]["inv_weight"]
        
        return weight, sldj

    def _is_eval_dict_empty(self, device_name=None):
        if device_name is not None:
            return not (device_name in self.eval_dict)
        else:
            return len(self.eval_dict) == 0

    def _empty_eval_dict(self, device_name=None):
        if device_name is not None:
            self.eval_dict.pop(device_name)
        else:
            self.eval_dict = defaultdict(lambda : self._get_default_inner_dict())
        
    def forward(self, x, ldj=None):
        if ldj is None:
            ldj = torch.zeros(x.shape[0], device=x.device)
        weight, sldj = self._get_weight(device_name=str(x.device), inverse=False)
        ldj = ldj - sldj
        z = torch.matmul(x, weight)
        return z, ldj

    def reverse(self, x):
        weight, sldj = self._get_weight(device_name=str(x.device), inverse=True)
        z = torch.matmul(x, weight)
        return z

class NoOpFlow(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, ldj=None):
        if ldj is None:
            ldj = x.new_zeros(x.shape[0], )
        return x, ldj

    def reverse(self, x):
        return x

def get_on_off_diag_means_for_tensor_dict(tensor_dict, unpack_batch=False):
    stacked_tensor = torch.stack(list(tensor_dict.values()), dim=2)  # batch x num_factors x num_factors
    # Get mean of diag elements, and mean of off-diag elements
    diag = torch.diagonal(stacked_tensor, dim1=1, dim2=2)
    if unpack_batch:
        diag = diag[diag.shape[0] // 2:]
    diag_mean = diag.mean()
    # zero out the diagonal elements
    only_offdiag = stacked_tensor * (
                1 - torch.eye(stacked_tensor.shape[1], device=stacked_tensor.device)[None])
    if unpack_batch:
        only_offdiag = only_offdiag[:only_offdiag.shape[0] // 2]
    only_offdiag_mean = only_offdiag.sum() / (
                only_offdiag.shape[0] * only_offdiag.shape[1] * (only_offdiag.shape[1] - 1))
    return diag_mean, only_offdiag_mean