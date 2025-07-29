import torch.nn.functional as F
import torch
import torch_dct as dct

from libs.utils import wrapToMax, channel_norm
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

import copy


def deep_denoiser(x, noise_level=50, model=None, data_range=1.0):

    if isinstance(noise_level, float):
        noise_level = torch.tensor([noise_level], device=x.device)

    noise_level = noise_level.reshape(-1, 1, 1, 1)

    noise_level_map = (
        torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        * (noise_level * data_range)
        / 255.0
    )

    x = torch.cat((x, noise_level_map), dim=1)
    x = model(x)

    return x


class Unwrapping(object):


    def __init__(self, y, mx):
        # Mdx_y = M( Delta_x @ y ) , Mdy_y = M( Delta_y @ y )
        Mdx_y = F.pad(
            wrapToMax(torch.diff(y, 1, dim=-1), mx), (1, 1, 0, 0), mode="constant"
            )
        
        Mdy_y = F.pad(
            wrapToMax(torch.diff(y, 1, dim=-2), mx), (0, 0, 1, 1), mode="constant"
        )

        # DTMDy = D^T ( Mdx_y, Mdy_y )
        DTMDy = -(
                torch.diff(Mdx_y, 1, dim=-1)
            +   torch.diff(Mdy_y, 1, dim=-2)
        )

        self.DTMDy = DTMDy
        self.mx = mx

    def forward(self, xtilde, epsilon):

        rho = self.DTMDy + (epsilon / 2) * xtilde
        dct_rho = dct.dct_2d(rho, norm="ortho")

        NX, MX = rho.shape[-1], rho.shape[-2]
        I, J = torch.meshgrid(torch.arange(0, MX), torch.arange(0, NX), indexing="ij")
        I, J = I.to(rho.device), J.to(rho.device)

        I, J = I.unsqueeze(0).unsqueeze(0), J.unsqueeze(0).unsqueeze(0)

        denom = 2 * (
            (epsilon / 4)
            + 2
            - (torch.cos(torch.pi * I / MX) + torch.cos(torch.pi * J / NX))
        )
        denom = denom.to(rho.device)

        dct_phi = dct_rho / denom
        dct_phi[..., 0, 0] = 0

        phi = dct.idct_2d(dct_phi, norm="ortho")
        return phi



def admm(
    y,
    denoiser,
    model,
    max_iters,
    mx=255,
    epsilon=1.0,
    _lambda=1,
    gamma=1,
    plot_iters=False,
):

    # initialize variables
    iters = 1
    unwrapping_fn = Unwrapping(y, mx)

    u_t = torch.zeros_like(y)

    # initialize x_t
    x_0 = unwrapping_fn.forward(u_t, 0.0)
    x_t = x_0

    # denoising step
    vtilde  = x_t + u_t
    sigma   = _lambda / epsilon
    vtilde  = torch.cat([vtilde, y], dim=1)  # Concatenate y for the denoiser
    v_t     = denoiser(vtilde, sigma, model)
    v_t     = torch.nn.functional.relu(v_t)  # Ensure non-negativity

    # update step
    u_t = u_t + x_t - v_t


    for i in range(max_iters):

        # inversion step
        xtilde  = v_t - u_t
        x_t     = unwrapping_fn.forward(xtilde, epsilon)

        # denoising step
        vtilde  = x_t + u_t
        sigma   = _lambda / epsilon
        vtilde  = torch.cat([vtilde, y], dim=1)  # Concatenate y for the denoiser
        v_t     = denoiser(vtilde, sigma, model)
        v_t     = torch.nn.functional.relu(v_t)  # Ensure non-negativity
        # v_t = vtilde

        # update step
        u_t = u_t + x_t - v_t
        epsilon = epsilon * gamma

    x_hat = v_t

    return x_hat 




class Unrolled(nn.Module):

    def __init__(self, model, denoiser, max_iters=5, gamma=1.01, epsilon=1e-3):

        super(Unrolled, self).__init__()

        self.denoiser = denoiser
        self.max_iters = max_iters

        self.gamma = gamma
        self.epsilon = epsilon
        self.model = model

    def freeze_model(self):

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, y, std):

        _lambda = std * self.epsilon
        gamma = self.gamma
        epsilon = self.epsilon

        # return y*_lambda + gamma
        out = admm(
            y,
            self.denoiser,
            self.model,
            self.max_iters,
            mx=1.0,
            gamma=gamma,
            epsilon=epsilon,
            _lambda=_lambda,
        )


        return out
