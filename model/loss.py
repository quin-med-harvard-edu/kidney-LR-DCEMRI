import torch
import numpy as np


def data_l2_loss(x: torch.tensor,
                 y: torch.tensor,
                 k_traj: torch.tensor,
                 sqrt_dcf: torch.tensor,
                 coil_p: torch.tensor,
                 smap: torch.tensor,
                 operator):
    [size_batch, _, n_sample, n_spk] = sqrt_dcf.shape
    num_coil = coil_p.shape[1]

    x_ = torch.complex(x[:, 0, :, :], x[:, 1, :, :])

    out_ = operator(x_.unsqueeze(1) * coil_p,
                    k_traj, smaps=smap)
    out_ = out_.reshape(size_batch, num_coil, n_sample, n_spk)
    out_ *= sqrt_dcf
    out_ = out_.permute(0, 2, 1, 3)

    return torch.mean((torch.real(out_) - torch.real(y)) ** 2 +
                      (torch.imag(out_) - torch.imag(y)) ** 2)
