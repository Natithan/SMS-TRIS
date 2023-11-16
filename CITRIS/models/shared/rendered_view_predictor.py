from experiments.utils import maybe_id_and_ood_unpack, coral_loss
import torch.nn as nn
from models.shared import log_dict
from models.shared.next_step_predictor import NextStepPredictor
import os
import torch.nn.functional as F
from collections import defaultdict
import torch
import numpy as np
import wandb

class RenderedViewPredictor(NextStepPredictor):

    DATA_SETTING = 'RVP'
    def __init__(self, DataClass, num_causal_vars, num_latents):
        super().__init__(DataClass, num_causal_vars)
        # if self.hparams.use_baseline_correlation_metrics:
        # Changed: now always log both baseline and non-baseline metrics
        self.register_buffer('last_target_assignment', torch.zeros(num_latents,
                                                                   # self.num_causal_vars,
                                                                   self.num_coarse_vars # For pong compatibility
                                                                   ))
        self._model = None


    def log_mse_losses(self, batch, dataloader_idx, split='val'):
        # x_mseloss, z_mseloss = self.get_x_mseloss(batch, also_z_mse=True)
        # maybe_cma_prefix = 'cma_' if  (dataloader_idx == 2) else ''
        # self.mylog(f'{maybe_cma_prefix}{split}_x_mseloss', x_mseloss)
        # self.mylog(f'{maybe_cma_prefix}{split}_z_mseloss', z_mseloss)
        pass # Redundant with FramePrediction callback, and also logging it through here gives a worse xmse, I couldn't figure out why, if I validate manually it gives a similar xmse though, so assuming the callback is correct


    def get_x_mseloss(self, batch, also_z_mse=False):
        xt, xt1, It1 = batch['imgs'][:, 0], batch['imgs'][:, 1], batch['targets']
        if also_z_mse:
            xt1_pred, z_mseloss = self.sample_xt1(xt, It1, also_z_mse=True, true_x_t1=xt1)
        else:
            xt1_pred = self.sample_xt1(xt, It1)
        mseloss = F.mse_loss(xt1_pred, xt1)
        if also_z_mse:
            return mseloss, z_mseloss
        return mseloss

    def sample_xt1(self, x_t, I_t1, also_z_mse=False, true_x_t1=None):
        z_t = self.encode(x_t,random=False)
        z_t1 = self.sample_zt1(z_t, I_t1)
        if also_z_mse:
            true_z_t1 = self.encode(true_x_t1,random=False)
            z_mse = F.mse_loss(z_t1, true_z_t1)
        x_t1 = self.decode(z_t1)
        if also_z_mse:
            return x_t1, z_mse
        return x_t1

    @classmethod
    def get_specific_cormet_args(cls, **kwargs):
        return {
            "mfp": False,
        }


    def validation_epoch_end(self, *args, **kwargs):
        # Logging at the end of validation
        super().validation_epoch_end(*args, **kwargs)
        if len(self.all_val_dists.keys()) > 0: # dict_keys(['pos-x', 'pos-y', 'pos-z', 'rot-alpha', 'rot-beta', 'rot-spot', 'hue-object', 'hue-spot', 'hue-back', 'obj-shape'])
            if self.current_epoch > 0:
                means = {}
                if 'object_indices' in self.all_val_dists:
                    obj_indices = torch.cat(self.all_val_dists['object_indices'], dim=0)
                    unique_objs = obj_indices.unique().detach().cpu().numpy().tolist()
                for key in self.all_val_dists:
                    if key == 'object_indices':
                        continue
                    vals = torch.cat(self.all_val_dists[key], dim=0)
                    if 'object_indices' in self.all_val_dists:
                        for o in unique_objs:
                            sub_vals = vals[obj_indices == o]
                            key_obj = key + f'_obj_{o}'
                            # self.logger.experiment.add_histogram(key_obj, sub_vals, self.current_epoch)
                            # check if any nans
                            if torch.isnan(vals).any():
                                print(f'WARNING: {key} has nans, skipping histogram')
                            else:
                                self.logger.experiment.log({key_obj: wandb.Histogram(sub_vals.cpu())}, step=self.global_step) # Nathan. Not sure if changing epoch to global step is needed, but seems to make more sense.
                            means[key_obj] = sub_vals.mean().item()
                    else:
                        # self.logger.experiment.add_histogram(key, vals, self.current_epoch)
                        # check if any nans
                        if torch.isnan(vals).any():
                            print(f'WARNING: {key} has nans, skipping histogram')
                        else:
                            self.logger.experiment.log({key: wandb.Histogram(vals.cpu())}, step=self.global_step) # Nathan. Not sure if changing epoch to global step is needed, but seems to make more sense.
                        means[key] = vals.mean().item()
                log_dict(d=means,
                         name='triplet_dists',
                         current_epoch=self.current_epoch,
                         log_dir=self.logger.experiment.dir)  # Nathan: logger.experiment.dir instead of logger.log_dir for wandb logger
            self.all_val_dists = defaultdict(list)
        if len(self.all_v_dicts) > 0:
            outputs = {}
            for key in self.all_v_dicts[0]:
                outputs[key] = torch.cat([v[key] for v in self.all_v_dicts], dim=0).cpu().numpy()
            np.savez_compressed(os.path.join(self.logger.experiment.dir, 'causal_encoder_v_dicts.npz'),
                                **outputs)  # Nathan: logger.experiment.dir instead of logger.log_dir for wandb logger
            self.all_v_dicts = []


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval_step(batch, batch_idx, dataloader_idx, stage='val')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval_step(batch, batch_idx, dataloader_idx, stage='test')

    def eval_step(self, batch, batch_idx, dataloader_idx=0, stage='val',only_cheating=False):
        # if dataloader_idx == 0:
        if batch['isTriplet'][0]:
            # imgs, *_ = batch
            if not only_cheating:
                loss_learned_target_assignment = self.triplet_evaluation(batch, mode=stage, dataloader_idx=dataloader_idx, use_cheating_target_assignment=False)
                self.mylog(f'{stage}_triplet_cnd', loss_learned_target_assignment)
            loss_cheating_target_assignment = self.triplet_evaluation(batch, mode=stage, dataloader_idx=dataloader_idx, use_cheating_target_assignment=True)
            self.mylog(f'{stage}_triplet_cnd_cheating', loss_cheating_target_assignment)
        else:
            self.log_mse_losses(batch, dataloader_idx, split=stage)

    # Nathan-added
    def decode(self, z):
        x = self.decoder(z)
        return x

    def sample_zt1(self,*args,**kwargs):
        raise NotImplementedError


    def get_losses(self, batch, batch_idx=None, args=None, pretrain_phase=False):
        '''
        Dimensions are marked as follows:
        - B: Batch size
        - L: Sequence length
        - S: Number of samples
            - S0: Number of samples for timestep t
            - S1: Number of samples for timestep t+1
        - Z: Latent dimension
        - Ch: Number of channels (3 for RGB)
        - H: Height
        - W: Width
        - F: number of true causal factors (dimensionality of c_t)
        '''
        id_and_ood = False
        batch, id_and_ood, id_batch_size, ood_batch_size = maybe_id_and_ood_unpack(batch, id_and_ood)
        loss_dict = {}
        # xtt1, It1, ctt1 = batch # It1: B_L-1_C
        xtt1, It1, ctt1 = batch['imgs'], batch['targets'], batch['lat']
        if id_and_ood:
            xtt1_to_encode = torch.cat([xtt1, batch['ood_imgs']], dim=0)
            It1_for_nontf = torch.cat([It1, batch['ood_targets']], dim=0)
            ctt1_for_nontf = torch.cat([ctt1, batch['ood_lat']], dim=0)
        else:
            xtt1_to_encode = xtt1
            It1_for_nontf = It1
            ctt1_for_nontf = ctt1
        batch_size_for_unexpand, seq_len = xtt1_to_encode.shape[:2]
        if seq_len > 2:
            raise NotImplementedError
        encoder_outputs = self.get_encoder_outputs(xtt1_to_encode)
        encoder_outputs_for_tf = self.get_encoder_outputs_for_tf(encoder_outputs, id_and_ood, id_batch_size)

        self.add_teacher_forced_losses(encoder_outputs_for_tf, It1, loss_dict, xtt1, args, grad_for_idloss=id_and_ood, pretrain_phase=pretrain_phase)
        ztt1_mean, ztt1_sample = self.get_unexpanded_encoder_outputs(encoder_outputs, batch_size_for_unexpand)


        zt_sample, zt1_sample = ztt1_sample[:, 0], ztt1_sample[:, 1]

        if id_and_ood:
            # self.add_coral_losses(pred_zt1_mean, pred_zt1_logstd, loss_dict, seq_len, id_batch_size, ood_batch_size)
            self.add_coral_losses(ztt1_sample, loss_dict, seq_len, id_batch_size, ood_batch_size)

        if not pretrain_phase:
            self.add_autoreg_losses(It1_for_nontf, ctt1_for_nontf, loss_dict, xtt1_to_encode,
                                    zt_sample, ztt1_mean, args)

        return loss_dict

    def add_autoreg_losses(self, It1_for_nontf, ctt1_for_nontf, loss_dict, xtt1_to_encode,
                           zt_sample, ztt1_mean, args):
        use_mean_to_autoregress_lst = self.get_use_mean_to_autoregress_lst(args)
        use_zt_mean_for_flp_input = True  # Could be that using zt_sample helps training in some way, but for now we use zt_mean
        zt_for_flp_input = ztt1_mean[:, 0] if use_zt_mean_for_flp_input else zt_sample
        for use_mean_to_autoregress in use_mean_to_autoregress_lst:
            postfix = '_mar' if use_mean_to_autoregress else '_sar' # for mean auto regress or sample auto regress
            with torch.no_grad():
                prior_sampled_zt1, pred_zt1_mean, pred_zt1_logstd = self.sample_zt1(zt_for_flp_input, It1_for_nontf,
                                                                                    use_mean_to_autoregress=use_mean_to_autoregress,
                                                                                    also_return_params=True)
                prior_sampled_xt1 = self.decode(prior_sampled_zt1)
            loss_dict[f'mse_x_mean{postfix}'] = F.mse_loss(prior_sampled_xt1, xtt1_to_encode[:, 1:].flatten(0, 1),
                                                 reduction='none').mean()
            loss_dict[f'mse_z_mean{postfix}'] = F.mse_loss(prior_sampled_zt1, ztt1_mean[:, 1:].flatten(0, 1), reduction='none').mean()
            loss_dict[f'mse_per_z{postfix}'] = F.mse_loss(prior_sampled_zt1, ztt1_mean[:, 1:].flatten(0, 1), reduction='none').mean(
                dim=0)
            with torch.no_grad():
                ct1_pred = self.causal_encoder(
                    prior_sampled_xt1)  # dict with F elements, each element B(L-1)S0_X, where X = 1 for continuous, 2 for angle, num_cats for categorical
            ct1_true = ctt1_for_nontf[:, 1:].flatten(0, 1)
            _, _, c_normed_dists = self.causal_encoder.calculate_loss_distance(ct1_pred, ct1_true)
            loss_dict[f'normed_dist_per_c{postfix}'] = torch.tensor([d.mean() for d in c_normed_dists.values()])
            loss_dict[f'normed_dist_c_mean{postfix}'] = loss_dict[f'normed_dist_per_c{postfix}'].mean()

    def get_use_mean_to_autoregress_lst(self, args):
        match args.use_mean_to_autoregress_str:
            case 'both':
                use_mean_to_autoregress_lst = [True, False]
            case 'True':
                use_mean_to_autoregress_lst = [True]
            case 'False':
                use_mean_to_autoregress_lst = [False]
            case _:
                raise ValueError()
        return use_mean_to_autoregress_lst

    def get_encoder_outputs_for_tf(self, encoder_outputs, id_and_ood, id_batch_size):
        if id_and_ood:
            encoder_outputs_for_tf = {k: v[:id_batch_size] for k, v in encoder_outputs.items()}
        else:
            encoder_outputs_for_tf = encoder_outputs
        return encoder_outputs_for_tf

    def get_unexpanded_encoder_outputs(self, encoder_outputs, batch_size):
        ztt1_mean, ztt1_sample = encoder_outputs['ztt1_mean'], encoder_outputs['ztt1_sample']
        return ztt1_mean, ztt1_sample

    # def add_coral_losses(self, pred_zt1_mean, pred_zt1_logstd, loss_dict, seq_len, id_batch_size, ood_batch_size):
    def add_coral_losses(self, ztt1_sample, loss_dict, seq_len, id_batch_size, ood_batch_size):
        # concat prior mean and logstd in z-dim
        # prior_params = torch.cat([pred_zt1_mean, pred_zt1_logstd],
        #                          dim=-1)  # (Bi+Bo)(L-1)_Z, (Bi+Bo)(L-1)_Z] ->  (Bi+Bo)(L-1)_(2*Z)
        prior_params = ztt1_sample.flatten(0,1)
        id_prior_params = prior_params[:id_batch_size * (seq_len - 1)]  # (Bi+Bo)(L-1)_(2*Z) -> Bi(L-1)_(2*Z)
        ood_prior_params = prior_params[id_batch_size * (seq_len - 1):]  # (Bi+Bo)(L-1)_(2*Z) -> Bo(L-1)_(2*Z)
        l_coral = coral_loss(id_prior_params, ood_prior_params)
        loss_dict['l_coral'] = l_coral
        # loss_dict['fsl_train_loss'] += l_coral * self.hparams.beta_coral
        loss_dict['ID_loss'] += l_coral * self.hparams.beta_coral

    def get_encoder_outputs(self, xtt1):
        # En- and decode every element of the sequence, except first element no decoding
        ztt1_mean, ztt1_logstd = self.encoder(xtt1.flatten(0, 1))
        ztt1_sample = ztt1_mean + torch.randn_like(ztt1_mean) * ztt1_logstd.exp()
        batch_size = xtt1.shape[0]
        # unflatten
        ztt1_sample = ztt1_sample.unflatten(0, (batch_size, -1))
        ztt1_mean = ztt1_mean.unflatten(0, (batch_size, -1))
        ztt1_logstd = ztt1_logstd.unflatten(0, (batch_size, -1))
        # return ztt1_logstd, ztt1_mean, ztt1_sample
        return {"ztt1_logstd": ztt1_logstd, "ztt1_mean": ztt1_mean, "ztt1_sample": ztt1_sample}


    def add_teacher_forced_losses(self, encoder_outputs, It1, loss_dict, xtt1, args=None, grad_for_idloss=False, pretrain_phase=False):
        batch_size = xtt1.shape[0]

        ztt1_logstd, ztt1_mean, ztt1_sample = encoder_outputs['ztt1_logstd'], encoder_outputs['ztt1_mean'], encoder_outputs['ztt1_sample']
        if self.hparams.use_flow_prior:
            raise NotImplementedError
        else:
            if pretrain_phase:
                overriden_fixed_logstd = False # hacky
            else:
                overriden_fixed_logstd = None
            kld_part = self.get_kld_part_and_log(It1, loss_dict, xtt1, ztt1_logstd, ztt1_mean, ztt1_sample,overriden_fixed_logstd=overriden_fixed_logstd)
        # Calculate reconstruction loss
        if isinstance(self.decoder, nn.Identity):
            raise NotImplementedError
            rec_loss = ztt1_mean.new_zeros(xtt1.shape[0], xtt1.shape[1])
        if any(p.requires_grad for p in self.decoder.parameters()):
            x_rec = self.get_x_rec(ztt1_sample)
            x_rec = x_rec.unflatten(0, (batch_size, -1))
            rec_loss_part = self.get_rec_loss_part_and_log(loss_dict, x_rec, xtt1)
        else:
            rec_loss_part = 0
        # Combine to full loss
        kld_factor = self.kld_scheduler.get_factor(self.global_step)
        per_element_loss = (kld_factor * kld_part + rec_loss_part)
        # if ood_batch_size > 0 and args.coral_upweight:
        #     id_loss = per_element_loss[:-ood_batch_size].mean()
        #     ood_loss = per_element_loss[-ood_batch_size:].mean()
        #     loss = (id_loss + ood_loss) / 2
        # else:
        #     loss = per_element_loss.mean()
        loss = per_element_loss.mean()
        loss = loss / (xtt1.shape[1] - 1)
        loss_dict['fsl_train_loss'] = loss # TODO maybe this should only be the kld_part, as the x_rec part doesn't affect FLP params
        loss_dict['ID_loss'] = loss.clone()
        self.maybe_add_intv_loss(loss, loss_dict, ztt1_sample, It1, args, grad_for_idloss)

    def maybe_add_intv_loss(self, *args, **kwargs):
        raise NotImplementedError

    def get_rec_loss_part_and_log(self, loss_dict, x_rec, xtt1):
        rec_loss = F.mse_loss(x_rec, xtt1[:, 1:], reduction='none').sum(dim=[-3, -2, -1])
        rec_loss_part = rec_loss.sum(dim=1)
        loss_dict['rec_loss_t1'] = rec_loss.mean() / (xtt1.shape[1] - 1)
        return rec_loss_part

    def get_kld_part_and_log(self, It1, loss_dict, xtt1, ztt1_logstd, ztt1_mean, ztt1_sample,overriden_fixed_logstd=None):
        kld = self.get_kld(It1, ztt1_logstd, ztt1_mean, ztt1_sample, overriden_fixed_logstd=overriden_fixed_logstd)
        kld_part = kld * self.hparams.beta_t1
        loss_dict['kld_t1'] = kld.mean() / (xtt1.shape[1] - 1)
        return kld_part

    def get_kld(self, *args, **kwargs):
        raise NotImplementedError

    def get_x_rec(self, ztt1_sample):
        # x_rec = self.decoder(ztt1_sample.unflatten(0, xtt1.shape[:2])[:, 1:].flatten(0, 1))  # x_{t+1}
        x_rec = self.decoder(ztt1_sample[:, 1:].flatten(0, 1))  # x_{t+1}
        return x_rec

    def _get_loss_batch_unpack(self, batch):
        # if len(batch) == 2:
        #     imgs, target = batch
        #     labels = imgs
        # else:
        #     imgs, labels, target = batch
        imgs = batch['imgs']
        target = batch['targets']
        if len(batch) == 3:
            labels = imgs
        else:
            raise NotImplementedError()
        return imgs, labels, target

    @property
    def ae_part(self):
        return self

    def fsl_calibration_metric(self, args):
        return 'mse_per_z' + ('_sar' if args.use_mean_to_autoregress_str == 'False' else '_mar')


    def prep_for_fsl_train(self):
        super().prep_for_fsl_train()
        self.ae_part.eval() # To prevent BatchNorm from taking its mean and std from the few-shot data