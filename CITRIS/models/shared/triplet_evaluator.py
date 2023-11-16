from collections import defaultdict
from util import tn, mylog
import pytorch_lightning as pl
import torch.nn as nn
import torch

class TripletEvaluator(pl.LightningModule):
    '''
    A superclass of AutoregNormalizingFlow and CITRISVAE whose only job is implementing triplet evaluation to avoid code duplication.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_v_dicts = []
        self.all_val_dists = defaultdict(list)

    def mylog(self, name, value):
        mylog(self, name, value)

    def triplet_evaluation(self, batch, mode='val', dataloader_idx=None, use_cheating_target_assignment=False):
        """ Evaluates the triplet prediction for a batch of images """
        # Input handling
        imgs, labels, latents, obj_indices, source = unpack_triplet_batch(batch)

        triplet_label = labels[:, -1]
        # Estimate triplet prediction
        triplet_rec = self.triplet_prediction(imgs, source, use_cheating_target_assignment)

        if self.causal_encoder is not None and latents is not None:
            self.causal_encoder.eval()
            # Evaluate the causal variables of the predicted output
            with torch.no_grad():
                losses, dists, norm_dists, v_dict = self.causal_encoder.get_distances(triplet_rec, latents[:, -1],
                                                                                      return_norm_dists=True,
                                                                                      return_v_dict=True)
                self.all_v_dicts.append(v_dict)
                rec_loss = sum([norm_dists[key].mean() for key in losses])
                mean_loss = sum([losses[key].mean() for key in losses])
                postfix = '_cheating' if use_cheating_target_assignment else ''
                self.mylog(f'{mode}_distance_loss{postfix}', mean_loss)
                for key in dists:
                    self.all_val_dists[key].append(dists[key])
                    self.mylog(f'per_cdim/{mode}_{key}_triplet_dist{postfix}', dists[key].mean())
                    self.mylog(f'per_cdim/{mode}_{key}_triplet_norm_dist{postfix}', norm_dists[key].mean())
                    if obj_indices is not None:  # For excluded object shapes, record results separately
                        for v in obj_indices.unique().detach().cpu().numpy():
                            self.mylog(f'{mode}_{key}_dist_obj_{v}{postfix}', dists[key][obj_indices == v].mean())
                            self.mylog(f'{mode}_{key}_norm_dist_obj_{v}{postfix}',
                                       norm_dists[key][obj_indices == v].mean())
                # self.mylog(f'{mode}_avg_triplet_dist{postfix}', [d.mean() for d in dists.values()]) # angles are between 0 and 180, so mean is not meaningful
                self.mylog(f'{mode}_avg_triplet_norm_dist{postfix}', torch.stack(list(norm_dists.values())).mean())
                if obj_indices is not None:
                    self.all_val_dists['object_indices'].append(obj_indices)
                if self.current_epoch > 0 and self.causal_encoder_true_epoch >= self.current_epoch:
                    self.causal_encoder_true_epoch = self.current_epoch
                    if len(triplet_label.shape) == 2 and hasattr(self, 'autoencoder'):
                        triplet_label = self.autoencoder.decoder(triplet_label)
                    _, true_dists = self.causal_encoder.get_distances(triplet_label, latents[:, -1])
                    for key in dists:
                        self.mylog(f'per_cdim/{mode}_{key}_true_dist{postfix}', true_dists[key].mean())
                    # self.mylog(f'{mode}_avg_true_dist{postfix}', [d.mean() for d in true_dists.values()])
                    self.mylog(f'{mode}_avg_true_norm_dist{postfix}', torch.stack(list(true_dists.values())).mean())
        else:
            rec_loss = torch.zeros(1, )

        return rec_loss




def flow_based_triplet_pred(ae, flow, num_latents, source, target_assignment, x_encs, z=None):
    assert (x_encs is None) != (z is None)
    if x_encs is not None:
        # Nathan: encode if needed
        if x_encs.shape[-1] != num_latents:
            x_encs = ae.encoder(x_encs.flatten(0, 1))[None]

        if isinstance(flow, nn.Identity):
            input_samples = flow(x_encs[:, :2].flatten(0, 1))
        else:
            input_samples, _ = flow(x_encs[:, :2].flatten(0, 1))
        input_samples = input_samples.unflatten(0, (-1, 2))
    else:
        input_samples = z
    # Map the causal mask to a latent variable mask
    if source.shape[-1] + 1 == target_assignment.shape[-1]:  # No-variables missing
        source = torch.cat([source, source[..., -1:] * 0.0], dim=-1)
    elif target_assignment.shape[-1] > source.shape[-1]:
        # raise NotImplementedError()
        assert (num_latents == 16) and (target_assignment.shape[-1] == 7) and (source.shape[-1] == 5), "Hacky way to guarantee that this is only allowed for Pong case"
        target_assignment = target_assignment[..., :source.shape[-1]] # TODO this is wrong for flow_layers!
    # Take the latent variables from encoding 1 respective to the mask, and encoding 2 the inverse
    mask_1 = (target_assignment[None, :, :] * (1 - source[:, None, :])).sum(dim=-1)
    mask_2 = 1 - mask_1
    triplet_samples = mask_1 * input_samples[:, 0] + mask_2 * input_samples[:, 1]
    # Decode by reversing the flow, and using the pretrained decoder
    ae.eval()  # Set to eval in any case
    triplet_samples = flow.reverse(triplet_samples)
    triplet_rec = ae.decoder(triplet_samples)
    return triplet_rec


def unpack_triplet_batch(batch):
    imgs = batch['encs'] if 'encs' in batch else batch['imgs']
    source = batch['targets']
    latents = batch['lat'] if 'lat' in batch else None
    obj_indices = batch['obj_triplet_indices'] if 'obj_triplet_indices' in batch else None
    labels = batch['labels'] if 'labels' in batch else imgs
    return imgs, labels, latents, obj_indices, source