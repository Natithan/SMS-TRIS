import torch

from pytorch_lightning.callbacks import LearningRateMonitor
from models.shared.callbacks import BaselineCorrelationMetricsLogCallback, FramePredictionCallback, FewShotOODCallback
from models.shared import SparsifyingGraphCallback, CorrelationMetricsLogCallback
from util import tn, mylog  # Not used, but often imported during debugging
import pytorch_lightning as pl

# Superclass of CITRISVAE and MixedFactorPredictor
from util import CUDAfy


class NextStepPredictor(pl.LightningModule):

    def __init__(self, DataClass, num_causal_vars):
        super().__init__()
        if num_causal_vars is None: # For backwards compatibility when MFP models didn't have this argument
            num_causal_vars = len(set(DataClass.FINE2COARSE.values()))
        self.num_coarse_vars = len(set(DataClass.FINE2COARSE.values()))
        self.short_factors = DataClass.SHORT_FACTORS
        self.num_causal_vars = num_causal_vars # size of intervention vector that is given in a batch. It decides how many factors / param_groups we'll try to disentangle
        self.num_groups = self.num_causal_vars + 1 # +1 for information never affected by intervention, that is hence assumed undisentangleable, called the 'trash' group
        self.num_fine_vars = len(DataClass.FINE2COARSE)
        self.fine_idx2coarse_idx = {DataClass.FACTORS.index(f): DataClass.COARSE_FACTORS.index(c) for f,c in DataClass.FINE2COARSE.items()}
        if self.num_causal_vars == self.num_coarse_vars:
            self.fine_idx2group_idx = self.fine_idx2coarse_idx
        elif self.num_causal_vars == self.num_fine_vars:
            self.fine_idx2group_idx = {i: i for i in range(self.num_fine_vars)}
        else:
            assert (DataClass.__name__ == "InterventionalPongDataset") and (self.num_causal_vars == 5) and (self.num_fine_vars == 7), "Hacky way of allowing Pong dataset to have 5 intvd factors and 2 never-intvd factors"
            self.fine_idx2group_idx = {
                0: 0, 1: 1, 2: 2, 3: 3, 4: 4, # 5 intvd factors
                5: 5, 6: 5 # 2 never-intvd factors, assigned to trash group
            }

    def get_sparse_update_params(self):
        raise NotImplementedError

    def get_grouped_flp_params(self):
        raise NotImplementedError

    def get_psi_params(self):
        raise NotImplementedError

    def default_use_maxcorr_psi(self):
        # Case 1: FV-RVP
        if hasattr(self.hparams, 'fullview_baseline') and self.hparams.fullview_baseline:
            return True
        # Case 2: IMFP:
        if hasattr(self, 'fixed_encoder') and self.fixed_encoder == 'identity':
            return True
        # Case 3: BMFP:
        if hasattr(self, 'fixed_encoder') and (self.fixed_encoder is None) and (self.get_psi_params().count_nonzero() == 0):
            return True
        return False

    def get_learnt_or_maxcorr_psi(self, use_maxcorr, also_return_updated_use_maxcorr=False):
        if use_maxcorr is None:
            use_maxcorr = self.default_use_maxcorr_psi()
        if use_maxcorr:
            result = self.last_target_assignment
        else:
            result = self.psi
        if also_return_updated_use_maxcorr:
            result = (result, use_maxcorr)
        return result

    def get_learnt_or_maxcorr_psi_no_trash(self, use_maxcorr):
        if use_maxcorr is None:
            use_maxcorr = self.default_use_maxcorr_psi()
        if use_maxcorr:
            return self.last_target_assignment
        else:
            return self.psi_no_trash

    def get_uf_idxs_for_uf_setting(self, unfreeze_setting, dataloader, distribution, args=None):
        setting = unfreeze_setting.split('uf_')[1]

        # In the case of FV-RVP, psi is not learnt, and hence useless. Instead, similarly to how correlation metrics are learned, we use the 'best match' that is found during baseline correlation metric learning
        # use_maxcorr = (hasattr(self.hparams, 'use_baseline_correlation_metrics') and self.hparams.use_baseline_correlation_metrics) or \
        #               (hasattr(self, 'fixed_encoder') and self.fixed_encoder == 'identity') or \
        #               args.use_maxcorr_psi
        # if args.use_maxcorr_psi is None:
        #     use_maxcorr = self.default_use_maxcorr_psi()
        # else:
        #     use_maxcorr = args.use_maxcorr_psi
        # if \
        #         (hasattr(self.hparams, 'use_baseline_correlation_metrics') and self.hparams.use_baseline_correlation_metrics) or \
        #         (hasattr(self, 'fixed_encoder') and self.fixed_encoder == 'identity') or \
        #                 args.use_maxcorr_psi\
        #         :
        #     psi = self.last_target_assignment
        #     psi_no_trash = self.last_target_assignment
        # else:
        #     psi = self.psi
        #     psi_no_trash = self.psi_no_trash
        psi = self.get_learnt_or_maxcorr_psi(args.use_maxcorr_psi)
        psi_no_trash = self.get_learnt_or_maxcorr_psi_no_trash(args.use_maxcorr_psi)
        # distribution is either 'ID', 'OOD_none', or 'OOD_{f1__f2 ...}' where fi are parts of  self.short_factors
        if distribution in ['ID', 'OOD_none']:
            return []
        ood_factors_fine = distribution.split('OOD_')[1].split('__')
        ood_idxs_fine = [self.short_factors.index(f) for f in ood_factors_fine]
        cheat_psi_idxs = [self.fine_idx2group_idx[idx] for idx in ood_idxs_fine]

        if setting == 'pred_pnu':
            with torch.no_grad():
                batch = CUDAfy(next(iter(dataloader)))
                loss_dict = self.get_losses(batch, args=args)
            new_metric = loss_dict[self.fsl_calibration_metric(args)]
            old_metric = self.id_loss
            relative_increase_per_latent = ((new_metric - old_metric)/old_metric) # vector with length num_latents
            relative_increase_per_factor = (relative_increase_per_latent @ psi_no_trash) / psi_no_trash.sum(0) # vector with length num_factors
            psi_idx = torch.argmax(relative_increase_per_factor.nan_to_num(float('-inf'))) # number between 0 and num_factors. If a factor has no latents, it will be nan. In that case, it will be ignored by argmax
            print(f"pred_pnu {psi_idx} matches cheat_pnu {cheat_psi_idxs}: {psi_idx in cheat_psi_idxs} for batch size "
                  # f"{batch[-1].shape[0]}")
                    f"{batch['lat'].shape[0]}")
            psi_idxs = [psi_idx]
        elif setting == 'cheat_pnu':
            psi_idxs = cheat_psi_idxs
        elif setting in ('all','all_enc'):
            psi_idxs = list(range(psi.shape[1]))
        # else if a digit, return that digit
        elif setting.isdigit():
            psi_idxs = [int(setting)]
        elif setting in self.short_factors:
            psi_idxs = [self.short_factors.index(setting)]
        else:
            raise ValueError(f"Unrecognized unfreeze setting: {unfreeze_setting}")
        unfreeze_idxs = torch.unique(torch.cat([torch.nonzero(psi[:, psi_idx]).flatten() for psi_idx in psi_idxs]))
        return unfreeze_idxs


        # psi_idx = self.hparams.DataClass.SHORT_COARSE_FACTORS_AND_TRASH.index(unfreeze_setting.split('uf_')[1]) if unfreeze_setting != 'uf_all' else 'uf_all' # number between 0 and num_factors

    def get_masks_for_unfreeze(self, unfreeze_setting, dataloader, distribution, args=None):
        unfreeze_idxs = self.get_uf_idxs_for_uf_setting(unfreeze_setting, dataloader, distribution, args)

        mask_dict = {}
        for name, param in self.get_sparse_update_params():
            if unfreeze_setting in ('uf_all','uf_all_enc'):
                param_mask = torch.ones_like(param)
            else:
                param_mask = torch.zeros_like(param)
                if not self.is_shared(name):
                    per_factor_dim = param.shape[0] // self.num_latents
                    for idx in unfreeze_idxs:
                        param_mask[idx * per_factor_dim:(idx + 1) * per_factor_dim] = 1
            mask_dict[name] = param_mask
        return mask_dict

    def prep_for_fsl_train(self):
        self.train()

    def is_shared(self):
        raise NotImplementedError

    @property
    def prior_parts(self):
        raise NotImplementedError

    def store_id_loss(self, loader, args):
        batch = CUDAfy(next(iter(loader)))
        with torch.no_grad():
            loss_dict = self.get_losses(batch, args=args)
        self.id_loss = loss_dict[self.fsl_calibration_metric(args)]

    @property
    def psi_no_trash(self):
        return self.psi[:, :-1]

    @classmethod
    def get_callbacks(cls, exmp_triplet_inputs=None, dataset=None, cluster=False, correlation_dataset=None, correlation_test_dataset=None, exmp_fp_inputs=None, mini=False, use_baseline_correlation_metrics=False,
                      fp_loader=None, fp_log_freq=1,
                      shift2loaders=None, ood_log_freq=1,
                      **kwargs):
        callbacks = []
        # Create learning rate callback
        callbacks.append(LearningRateMonitor('step'))
        if kwargs['add_enco_sparsification']:
            callbacks.append(SparsifyingGraphCallback(dataset=dataset, cluster=cluster,mini=mini)) # Nathan: added this
        if not kwargs['no_alignment_measuring']:
            callbacks += cls.add_cormet_callbacks(cluster, correlation_dataset, correlation_test_dataset, mini,
                                                  use_baseline_correlation_metrics)
        callbacks += cls.get_specific_callbacks(exmp_triplet_inputs=exmp_triplet_inputs, exmp_fp_inputs=exmp_fp_inputs, dataset=dataset, cluster=cluster, **kwargs)
        callbacks.append(FramePredictionCallback(fp_loader, mini=mini, every_n_epochs=fp_log_freq, **kwargs))
        # FewShotOODCallback
        if shift2loaders is not None:
            callbacks.append(FewShotOODCallback(shift2loaders, ood_log_freq, **kwargs))
        return callbacks

    @classmethod
    def add_cormet_callbacks(cls, cluster, correlation_dataset, correlation_test_dataset, mini,
                             use_baseline_correlation_metrics
                             ):
        new_callbacks = []
        cormet_args = {
            "dataset": correlation_dataset,
            "cluster": cluster,
            "test_dataset": correlation_test_dataset,
        }
        if mini:
            cormet_args['num_train_epochs'] = 5
        # cormet_args |= cls.get_specific_cormet_args(use_baseline_correlation_metrics)
        cormet_args |= cls.get_specific_cormet_args()
        # cormet_callback_class = BaselineCorrelationMetricsLogCallback if use_baseline_correlation_metrics else CorrelationMetricsLogCallback
        # new_callbacks.append(cormet_callback_class(**cormet_args))
        # Log both
        new_callbacks.append(CorrelationMetricsLogCallback(**cormet_args))
        new_callbacks.append(BaselineCorrelationMetricsLogCallback(**(cormet_args | {"ignore_learnt_psi": True})))
        return new_callbacks


    def mylog(self, name, value):
        mylog(self, name, value)