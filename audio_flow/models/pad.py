import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def pad1d(x: Tensor, patch_size: int) -> tuple[Tensor, int]:
    r"""Pad a tensor along the last dim.

    Args:
        x: (b, c, t)
    
    Outpus:
        out: (b, c, t')
    """

    T = x.shape[2]
    patch_size_t = patch_size
    pad_t = math.ceil(T / patch_size_t) * patch_size_t - T
    x = F.pad(x, pad=(0, pad_t))

    return x, pad_t

def unpad1d(x: Tensor, pad_t: int) -> Tensor:
    r"""Unpad a tensor to the original shape.

    Args:
        x: (b, c, t')
    
    Outpus:
        out: (b, c, t)
    """

    if pad_t != 0:
        x = x[:, :, 0 : -pad_t, :]

    return x


def pad2d(x: Tensor, patch_size: tuple[int, int]) -> tuple[Tensor, int, int]:
    r"""Pad a tensor along the last two dims.

    Args:
        x: (b, c, t, f)
    
    Outpus:
        out: (b, c, t', f')
    """

    T, F_ = x.shape[2:]
    patch_t, patch_f = patch_size

    pad_t = math.ceil(T / patch_t) * patch_t - T
    pad_f = math.ceil(F_ / patch_f) * patch_f - F_
    
    x = F.pad(x, pad=(0, pad_f, 0, pad_t))

    return x, pad_t, pad_f


def unpad2d(x: Tensor, pad_t: int, pad_f) -> Tensor:
    r"""Unpad a tensor to the original shape.

    Args:
        x: (b, c, t', f')
    
    Outpus:
        x: (b, c, t, f)
    """

    if pad_t != 0:
        x = x[:, :, 0 : -pad_t, :]

    if pad_f != 0:
        x = x[:, :, :, 0 : -pad_f]

    return x