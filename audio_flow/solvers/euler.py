from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def euler_solver(
    model: nn.Module, 
    noise: Tensor, 
    controls: dict, 
    n_steps: int,
) -> Tensor:

    t = torch.linspace(0, 1, n_steps, device=noise.device)
    x = noise
    
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        dx = model(t[i], x, controls)   # f(t, x)
        x = x + dt * dx              # Euler update

    return x