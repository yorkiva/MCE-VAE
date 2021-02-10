import os

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F



class NonLinear(nn.Module):
    def __init__(self, in_size, out_size, bias=True, activation=None):
        super(NonLinear, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(int(in_size), int(out_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

class GatedDense(nn.Module):
    def __init__(self, in_size, out_size, activation=None):
        super(GatedDense, self).__init__()
        self.activation = activation
        self.l_1 = nn.Linear(in_size, out_size)
        self.l_2 = nn.Linear(in_size, out_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.l_1(x)
        if self.activation is not None:
            h = self.activation(h)
        g = self.sigmoid(self.l_2(x))
        return h * g


# """
# def compute_mmd(z, reg_weight):
#     prior_z = torch.randn_like(z)
#     prior_z__kernel = self.compute_kernel(prior_z, prior_z)
#     z__kernel = self.compute_kernel(z, z)
#     priorz_z__kernel = self.compute_kernel(prior_z, z)

#     mmd = reg_weight * prior_z__kernel.mean() + reg_weight * z__kernel.mean() - 2 * reg_weight * priorz_z__kernel.mean()
#     return mmd

# def compute_kernel(x1, x2):
#     # Convert the tensors into row and column vectors
#     D = x1.size(1)
#     N = x1.size(0)

#     x1 = x1.unsqueeze(-2)  # Make it into a column tensor
#     x2 = x2.unsqueeze(-3)  # Make it into a row tensor
#     x1 = x1.expand(N, N, D)
#     x2 = x2.expand(N, N, D)
#     result = self.compute_inv_mult_quad(x1, x2)
#     return result"""

# def compute_inv_mult_quad(x1, x2, eps: float = 1e-7):
#     z_dim = x2.size(-1)
#     C = 2 * z_dim * 2.
#     kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

#     # Exclude diagonal elements
#     result = kernel.sum() - kernel.diag().sum()
#     return result

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch
