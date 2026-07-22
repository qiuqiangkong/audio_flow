import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate
from copy import deepcopy
from functools import partial

from audioflow.guidance.cfg import cfg_drop, cfg_forward
from audioflow.utils.torch import to_device


@torch.inference_mode()
def sample_latent(
    model: nn.Module, 
    noise: Tensor, 
    data: dict, 
    solver: object, 
    cfg_scale: float | None
) -> Tensor:

    device = noise.device
    model.eval()

    data_c = to_device(deepcopy(data), device)
    
    if cfg_scale:
        data_u = to_device(deepcopy(data), device)
        data_u = cfg_drop(data_u, dropout_prob=1.0)

        fn = partial(
            cfg_forward,  # 6 args: fn(model, t, x, data_c, data_u, cfg_scale)
            model=model,
            data_c=data_c,
            data_u=data_u,
            cfg_scale=cfg_scale
        )  # New function: fn(t, x)
    else:
        fn = partial(
            model,  # 3 args: fn(model, t, x, data)
            data=data_c
        )  # New function: fn(t, x)

    x_gen = solver(fn, noise, n_steps=100)  # (b, l, d)
    return x_gen
