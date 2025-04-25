import torch


def build_rope(
    seq_len: int, head_dim: int, base: int = 10000
) -> torch.Tensor:
    r"""Rotary Position Embedding.
    Modified from: https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py

    Args:
        seq_len: int, e.g., 1024
        head_dim: head dim, e.g., 768/24
        base: int

    Outputs:
        cache: (t, head_dim/2, 2)
    """
    
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))

    seq_idx = torch.arange(seq_len)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)