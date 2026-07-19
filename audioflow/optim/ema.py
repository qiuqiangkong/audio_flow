import torch.nn as nn
import torch


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float = 0.999):
    r"""Update EMA model weights and buffers from model."""

    for e, m in zip(ema.parameters(), model.parameters()):
        e.lerp_(m.detach(), 1 - decay)

    # Buffers (BN running stats, etc)
    for e, m in zip(ema.buffers(), model.buffers()):
        e.copy_(m)