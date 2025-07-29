import torch
import torch.nn.functional as F
import torch_dct as dct

from libs.utils import wrapToMax


def hard_threshold(x, threshold):
    return torch.where(torch.abs(x) > threshold, x, torch.zeros_like(x))


def spud(y, mx, threshold):

    # Mdx_y = M( Delta_x @ y ) , Mdy_y = M( Delta_y @ y )

    Mdx_y = torch.diff(y, 1, dim=-1)
    Mdy_y = torch.diff(y, 1, dim=-2)

    Mdx_y = F.pad(wrapToMax(Mdx_y, mx), (1, 0), mode="constant")
    Mdy_y = F.pad(wrapToMax(Mdy_y, mx), (0, 0, 1, 0), mode="constant")

    # DTMDy = D^T ( Mdx_y, Mdy_y )
    rho = -(
        torch.diff(F.pad(Mdx_y, (0, 1)), 1, dim=-1)
        + torch.diff(F.pad(Mdy_y, (0, 0, 0, 1)), 1, dim=-2)
    )

    dct_rho = dct.dct_2d(rho, norm="ortho")

    NX, MX = rho.shape[-1], rho.shape[-2]
    I, J = torch.meshgrid(torch.arange(0, MX), torch.arange(0, NX), indexing="ij")
    I, J = I.to(rho.device), J.to(rho.device)

    I, J = I.unsqueeze(0).unsqueeze(0), J.unsqueeze(0).unsqueeze(0)

    denom = 2 * (2 - (torch.cos(torch.pi * I / MX) + torch.cos(torch.pi * J / NX)))
    denom = denom.to(rho.device)

    dct_phi = dct_rho / denom
    dct_phi[..., 0, 0] = 0

    dct_phi = hard_threshold(dct_phi, threshold)

    phi = dct.idct_2d(dct_phi, norm="ortho")
    return phi
