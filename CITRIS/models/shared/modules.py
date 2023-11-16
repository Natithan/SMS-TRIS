import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import nn as nn
from torch.nn import functional as F


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """ Learning rate scheduler with Cosine annealing and warmup """

    def __init__(self, optimizer, warmup, max_iters, min_factor=0.05, offset=0):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_factor = min_factor
        self.offset = offset
        super().__init__(optimizer)
        if isinstance(self.warmup, list) and not isinstance(self.offset, list):
            self.offset = [self.offset for _ in self.warmup]

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        if isinstance(lr_factor, list):
            return [base_lr * f for base_lr, f in zip(self.base_lrs, lr_factor)]
        else:
            return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        lr_factor = lr_factor * (1 - self.min_factor) + self.min_factor
        if isinstance(self.warmup, list):
            new_lr_factor = []
            for o, w in zip(self.offset, self.warmup):
                e = max(0, epoch - o)
                l = lr_factor * ((e * 1.0 / w) if e <= w and w > 0 else 1)
                new_lr_factor.append(l)
            lr_factor = new_lr_factor
        else:
            epoch = max(0, epoch - self.offset)
            if epoch <= self.warmup and self.warmup > 0:
                lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class SineWarmupScheduler(object):
    """ Warmup scheduler used for KL divergence, if chosen """

    def __init__(self, warmup, start_factor=0.1, end_factor=1.0, offset=0):
        super().__init__()
        self.warmup = warmup
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.offset = offset

    def get_factor(self, step):
        step = step - self.offset
        if step >= self.warmup:
            return self.end_factor
        elif step < 0:
            return self.start_factor
        else:
            v = self.start_factor + (self.end_factor - self.start_factor) * 0.5 * (1 - np.cos(np.pi * step / self.warmup))
            return v


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, mask=None):
        super().__init__(in_features, out_features, bias)

        self.register_buffer('mask', mask)
    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def extra_repr(self):
        return super().extra_repr() + ', mask={}'.format(self.mask is not None)


class MultivarLinear(nn.Module):

    def __init__(self, input_dims, output_dims, extra_dims, bias=True, nokaiming=False):
        """
        Linear layer, which effectively applies N independent linear layers in parallel.

        Parameters
        ----------
        input_dims : int
                     Number of input dimensions per network.
        output_dims : int
                      Number of output dimensions per network.
        extra_dims : list[int]
                     Number of networks to apply in parallel. Can have multiple dimensions if needed.
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.extra_dims = extra_dims

        self.weight = nn.Parameter(torch.zeros(*extra_dims, output_dims, input_dims))
        if bias:
            self.bias = nn.Parameter(torch.zeros(*extra_dims, output_dims))
        if not nokaiming:
            nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        self.input_mask = None
        self.use_nonzero_mask = False

    def forward(self, x):
        # Shape preparation
        x_extra_dims = x.shape[1:-1]
        if len(x_extra_dims) > 0:
            for i in range(len(x_extra_dims)):
                assert x_extra_dims[-(i+1)] == self.extra_dims[-(i+1)], \
                    "Shape mismatch: X=%s, Layer=%s" % (str(x.shape), str(self.extra_dims))
        for _ in range(len(self.extra_dims)-len(x_extra_dims)):
            x = x.unsqueeze(dim=1)


        # Maybe mask (for parent-only)
        if self.input_mask is not None:
            # # put input mask on same device as weight (if not already)
            # if self.input_mask.device != weight.device:
            #     self.input_mask = self.input_mask.to(weight.device)
            # # weight: [1 (Batch), *Extra (typically # of factors), Output, Input]
            # # input_mask: [*Extra (typically # of factors), Input]
            # weight = weight * self.input_mask.unsqueeze(dim=0).unsqueeze(dim=-2)

            # mask the input instead of the weight
            # first make the Extra dimension of x the same as the extra dimension of the mask
            x = x.expand(-1, self.input_mask.shape[0], -1)
            # then mask the input
            if self.input_mask.device != x.device:
                self.input_mask = self.input_mask.to(x.device)
            x = x * self.input_mask # + (1 - self.input_mask) add 1 to the masked values so the corresponding weights can still contribute to the output
            if self.use_nonzero_mask:
                x = x + (1 - self.input_mask) # add 1 to the masked values so the corresponding weights can still contribute to the output

        # Unsqueeze
        x = x.unsqueeze(dim=-1) # [Batch, Extra (typically # of factors), Input, 1]
        weight = self.weight.unsqueeze(dim=0) # [1 (Batch), *Extra (typically # of factors), Output, Input]

        # Linear layer
        out = torch.matmul(weight, x).squeeze(dim=-1)
        
        # Bias
        if hasattr(self, 'bias'):
            bias = self.bias.unsqueeze(dim=0)
            out = out + bias
        return out

class MultivarMaskedLinear(MultivarLinear):
    def __init__(self, input_dims, output_dims, extra_dims, bias=True, mask=None, **kwargs):
        super().__init__(input_dims, output_dims, extra_dims, bias, **kwargs)
        assert mask is not None
        assert mask.shape == self.weight.shape
        self.register_buffer('mask', mask)

    def forward(self, input):
        # Shape preparation
        x_extra_dims = input.shape[1:-1]
        if len(x_extra_dims) > 0:
            for i in range(len(x_extra_dims)):
                assert x_extra_dims[-(i+1)] == self.extra_dims[-(i+1)], \
                    "Shape mismatch: X=%s, Layer=%s" % (str(input.shape), str(self.extra_dims))
        for _ in range(len(self.extra_dims)-len(x_extra_dims)):
            input = input.unsqueeze(dim=1)

        # Unsqueeze
        input = input.unsqueeze(dim=-1)
        weight = self.weight.unsqueeze(dim=0) * self.mask.unsqueeze(dim=0)

        # Linear layer
        out = torch.matmul(weight, input).squeeze(dim=-1)

        # Bias
        if hasattr(self, 'bias'):
            bias = self.bias.unsqueeze(dim=0)
            out = out + bias
        return out


    def extra_repr(self):
        return super().extra_repr() + ', mask={}'.format(self.mask is not None)


class MultivarLayerNorm(nn.Module):

    def __init__(self, input_dims, extra_dims):
        """
        Normalization layer with the same properties as MultivarLinear. 

        Parameters
        ----------
        input_dims : int
                     Number of input dimensions per network.
        extra_dims : list[int]
                     Number of networks to apply in parallel. Can have multiple dimensions if needed.
        """
        super().__init__()
        self.input_dims = input_dims
        self.extra_dims = np.prod(extra_dims)

        self.norm = nn.GroupNorm(self.extra_dims, self.input_dims * self.extra_dims)

    def forward(self, x):
        shape = x.shape
        out = self.norm(x.flatten(1, -1))
        out = out.reshape(shape)
        return out


class MultivarStableTanh(nn.Module):

    def __init__(self, input_dims, extra_dims, init_bias=0.0):
        """
        Stabilizing Tanh layer like in flows.

        Parameters
        ----------
        input_dims : int
                     Number of input dimensions per network.
        extra_dims : list[int]
                     Number of networks to apply in parallel. Can have multiple dimensions if needed.
        """
        super().__init__()
        self.input_dims = input_dims
        self.extra_dims = np.prod(extra_dims)

        self.scale_factors = nn.Parameter(torch.zeros(self.extra_dims, self.input_dims).fill_(init_bias))

    def forward(self, x):
        sc = torch.exp(self.scale_factors)[None]
        out = torch.tanh(x / sc) * sc
        return out


class AutoregLinear(nn.Module):

    def __init__(self, num_vars, inp_per_var, out_per_var, diagonal=False, 
                       no_act_fn_init=False, 
                       init_std_factor=1.0, 
                       init_bias_factor=1.0,
                       init_first_block_zeros=False,
                       no_interlatent_communication=False):
        """
        Autoregressive linear layer, where the weight matrix is correspondingly masked.

        Parameters
        ----------
        num_vars : int
                   Number of autoregressive variables/steps.
        inp_per_var : int
                      Number of inputs per autoregressive variable.
        out_per_var : int
                      Number of outputs per autoregressvie variable.
        diagonal : bool
                   If True, the n-th output depends on the n-th input (and all previous inputs).
                   If False, the n-th output only depends on the inputs 1 to n-1
        """
        super().__init__()
        self.linear = nn.Linear(num_vars * inp_per_var, 
                                num_vars * out_per_var)
        mask = torch.zeros_like(self.linear.weight.data)
        init_kwargs = {}
        if no_act_fn_init:  # Set kaiming to init for linear act fn
            init_kwargs['nonlinearity'] = 'leaky_relu'
            init_kwargs['a'] = 1.0
        for out_var in range(num_vars):
            out_start_idx = out_var * out_per_var
            out_end_idx = (out_var+1) * out_per_var
            inp_end_idx = (out_var+(1 if diagonal else 0)) * inp_per_var
            inp_start_idx = 0 if not no_interlatent_communication else out_var * inp_per_var
            if inp_end_idx > 0:
                mask[out_start_idx:out_end_idx, inp_start_idx:inp_end_idx] = 1.0
                if out_var == 0 and init_first_block_zeros:
                    self.linear.weight.data[out_start_idx:out_end_idx, inp_start_idx:inp_end_idx].fill_(0.0)
                else:
                    nn.init.kaiming_uniform_(self.linear.weight.data[out_start_idx:out_end_idx, inp_start_idx:inp_end_idx], **init_kwargs) # See https://arxiv.org/pdf/1502.01852.pdf#page=4
        self.linear.weight.data.mul_(init_std_factor)
        self.linear.bias.data.mul_(init_bias_factor)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)


class TanhScaled(nn.Module):
    """ Tanh activation function with scaling factor """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        assert self.scale > 0, 'Only positive scales allowed.'

    def forward(self, x):
        return torch.tanh(x / self.scale) * self.scale


