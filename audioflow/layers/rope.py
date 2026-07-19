import torch
import torch.nn as nn
from einops import rearrange
from torch import LongTensor, Tensor


def build_rope(
    head_dim: int, 
    pos=torch.arange(8192), 
    base=10000,
    device=None,
) -> Tensor:
    r"""Build RoPE matrix.

    l: seq_len
    h: head_dim
    
    Returns:
        rope: (l, h/2)
    """
    device = pos.device
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device) / head_dim))  # (h/2,)
    freq = torch.outer(pos, theta)  # (l, h/2)
    rope = torch.polar(torch.ones_like(freq), freq)  # (l, h/2), complex
    return rope


def apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    r"""Apply RoPE to data.

    b: batch_size
    l: seq_len
    n: n_heads
    h: head_dim

    Args:
        x: (b, l, n, h)
        rope: (l, h/2)

    Returns:
        out: (b, l, n, h)
    """
    x = rearrange(x, 'b l n (h c) -> b l n h c', c=2)  # (b, l, n, h/2, 2)
    x = torch.view_as_complex(x)  # (b, l, n, h/2), complex
    x = x * rope[None, 0 : x.shape[1], None, :]  # (b, l, n, h/2), complex
    x = torch.view_as_real(x)  # (b, l, n, h/2, 2)
    return rearrange(x, 'b l n h c -> b l n (h c)')  # (b, l, n, h)


def _plot(x, path):
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].matshow(x.real.cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(x.imag.cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig(path)
    print(f"(l, h) = {x.cpu().numpy().shape}")
    print(f"write out to {path}")


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    B = 4  # batch_size
    N = 8  # n_head
    H = 48  # head_dim
    D = N * H  # dim=384

    print("--- Example 1: RoPE (1D) ---")
    x = torch.rand((B, 100, N, H))  # (b, l, n, h)
    rope = build_rope(head_dim=H)  # (l, h/2)
    out = apply_rope(x, rope)  # (b, l, n, h)
    _plot(rope, "_1.pdf")

    print("--- Example 3: RoPE (2D Image) ---")
    x = torch.rand((B, 20, N, H))  # (b, l, n, h)
    h = torch.arange(4)
    w = torch.arange(5)
    poses = torch.meshgrid(h, w, indexing='ij')
    rope = torch.cat([build_rope(H // len(poses), pos.flatten()) for pos in poses], dim=-1)
    out = apply_rope(x, rope)
    _plot(rope, "_2.pdf")

    print("--- Example 3: RoPE (3D Video) ---")
    x = torch.rand((B, 60, N, H))  # (b, l, n, h)
    t = torch.arange(3)
    h = torch.arange(4)
    w = torch.arange(5)
    poses = torch.meshgrid(t, h, w, indexing='ij')
    rope = torch.cat([build_rope(H // len(poses), pos.flatten()) for pos in poses], dim=-1)
    out = apply_rope(x, rope)
    _plot(rope, "_3.pdf")
    