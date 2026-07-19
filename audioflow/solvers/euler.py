import torch
from torch import Tensor


def euler_solver(
    fn: callable,
    noise: Tensor, 
    n_steps: int,
) -> Tensor:

    t = torch.linspace(0, 1, n_steps, device=noise.device)
    x = noise
    
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        dx = fn(t=t[i], x=x)
        x = x + dt * dx

    return x