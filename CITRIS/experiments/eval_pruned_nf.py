import pickle
import matplotlib; matplotlib.use('TkAgg')
from CITRIS.models.citris_nf import *
from experiments.utils import *
from util import plot_tensor_as_img, tn
from constants import P2_ROOT
from models.shared.enco import ENCOGraphLearning

# def trained_prior(I_t1, pl_module, z_t):
#     prior = pl_module.prior_t1
#     psi = prior.get_target_assignment(hard=True)
#     zt1_intervened = prior._get_intv_params(z_t.shape, target=None)
#     zt1_intv_s = zt1_intervened[0] + zt1_intervened[1] * torch.randn(*zt1_intervened[1].shape)
#     zt1_intv_s_m = zt1_intv_s * I_t1 * psi[:, :-1]
#     zt1_intv_sms = zt1_intv_s_m.sum(dim=-1)
#     zt1_natural = prior._get_prior_params(z_t)
#     zt1_nat_s = zt1_natural[0] + zt1_natural[1] * torch.randn(*zt1_natural[1].shape)
#     mask = (I_t1 * psi[:, :-1]).sum(dim=-1).to(torch.bool)
#     zt1_comb = torch.where(mask, zt1_intv_sms, zt1_nat_s)
#     return zt1_comb
#
# def pruned_prior(I_t1, enco, z_t, temp_adj_matrix):
#     device = next(enco.net.parameters()).device
#     latent_mask0 = (enco.target_assignment[None, :, :, None] * temp_adj_matrix[None, None, :enco.num_blocks, :].expand(z_t.shape[0],-1,-1,-1)).sum(dim=-2)
#     latent_mask0 = latent_mask0.transpose(-2, -1)  # i -> j => Transpose to j <- i. [B x Z x C] -> [B x C x Z]
#
#     # inp[_,c,z_0], for z_0 in range(num_latents), is 0 if z_0 isn't assigned by psi to any of the c0 that cause c, and z0[z_0] otherwise
#     inp = torch.cat([
#         z_t[:, None] * latent_mask0,
#         latent_mask0 * 2 - 1,
#     ], dim=-1).to(device)
#
#     prior_mean, prior_logstd = enco.net(inp).chunk(2, dim=-1)
#     zt1_s_per_c = prior_mean + prior_logstd * torch.randn(*prior_logstd.shape,device=device)
#     zt1_s = (zt1_s_per_c * enco.target_assignment.transpose(0, 1)[None]).sum(dim=-2)
#     return zt1_s

def main():
    device = 'cpu'
    fp = f"{P2_ROOT}/CITRIS/experiments/checkpoints/CITRISNF/enco_net.pkl"
    with open(fp,'rb') as f:
        d = pickle.load(f)
    pl_module = CITRISNF.load_from_checkpoint(d['pl_module_path'],map_location={f'cuda:{i}':device for i in range(4)})
    enco_prior = d['enco_prior'].to(device)
    enco = ENCOGraphLearning(pl_module)
    enco.net = enco_prior
    temp_adj_matrix = d['temp_adj_matrix']
    dataset_args = Namespace(
        seed=42,
        data_dir="/cw/liir_data/NoCsBack/TRIS/causal3d_time_dep_all7_conthue_01_coarse/",
        coarse_vars=True,exclude_vars=None,exclude_objects=None,mini=True,seq_len=2,batch_size=5,num_workers=0,
        return_latents_train=True)
    datasets, data_loaders, data_name = load_datasets(dataset_args)
    batch = next(iter(data_loaders['train']))
    get_eval_for_batch(batch, device, enco, pl_module, temp_adj_matrix)


def get_eval_for_batch(batch, device, enco, pl_module, temp_adj_matrix):
    x_t, x_t1, I_t1, C_t, C_t1 = batch[0][:, 0], batch[0][:, 1], batch[1], batch[2][:,0], batch[2][:,1]
    z_t = pl_module.encode(x_t)
    batch_size = z_t.shape[0]
    # region get priors
    citris_prior = pl_module.prior_t1
    psi = citris_prior.get_target_assignment(hard=True)
    psi_intv = psi[:, :-1]
    psi_never_intv = psi[:, -1].to(torch.bool)
    zt1_intv_mean, zt1_intv_logstd = citris_prior._get_intv_params(z_t.shape, target=None)
    zt1_intv_s = zt1_intv_mean + zt1_intv_logstd * torch.randn(*zt1_intv_logstd.shape)
    zt1_intv_s_m = zt1_intv_s * I_t1 * psi_intv
    zt1_intv_sms = zt1_intv_s_m.sum(dim=-1)
    zt1_nat_mean, zt1_nat_logstd = citris_prior._get_prior_params(z_t)
    zt1_nat_s = zt1_nat_mean + zt1_nat_logstd * torch.randn(*zt1_nat_logstd.shape)
    intv_z_mask = (I_t1 * psi[:, :-1]).sum(dim=-1).to(torch.bool)
    unpruned_zt1 = torch.where(intv_z_mask, zt1_intv_sms, zt1_nat_s)
    # unpruned_zt1 = pl_module.sample_zt1(z_t, I_t1)

    latent_mask0 = (
                psi_intv[None, :, :, None] * temp_adj_matrix[None, None, :enco.num_blocks, :].expand(batch_size, -1, -1,
                                                                                                     -1)).sum(dim=-2)
    latent_mask0 = latent_mask0.transpose(-2, -1)  # i -> j => Transpose to j <- i. [B x Z x C] -> [B x C x Z]
    # inp[_,c,z_0], for z_0 in range(num_latents), is 0 if z_0 isn't assigned by psi to any of the c0 that cause c, and z0[z_0] otherwise
    inp = torch.cat([
        z_t[:, None] * latent_mask0,
        latent_mask0 * 2 - 1,
    ], dim=-1).to(device)
    zt1_enco_mean, zt1_enco_logstd = enco.net(inp).chunk(2, dim=-1)
    zt1_enco_s_per_c = zt1_enco_mean + zt1_enco_logstd * torch.randn(*zt1_enco_logstd.shape, device=device)
    zt1_enco_s = (zt1_enco_s_per_c * enco.target_assignment.transpose(0, 1)[None]).sum(dim=-2)
    pruned_zt1 = torch.where(
        intv_z_mask,
        zt1_intv_sms,
        torch.where(psi_never_intv[None].expand(batch_size, -1), zt1_nat_s, zt1_enco_s))
    # endregion
    unpruned_xt1 = pl_module.autoencoder.decoder(pl_module.flow.reverse(unpruned_zt1))
    pruned_xt1 = pl_module.autoencoder.decoder(pl_module.flow.reverse(pruned_zt1))
    predicted_C_t1s = [pl_module.causal_encoder.predict_causal_vars(x) for x in [x_t1, pruned_xt1, unpruned_xt1]]
    plot_tensor_as_img(
        torch.concat([x.permute(1, 2, 0, 3).flatten(-2, -1) for x in (x_t1, pruned_xt1, unpruned_xt1)], dim=1))


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == '__main__':
    main()