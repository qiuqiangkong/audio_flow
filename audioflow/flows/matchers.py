# Modified from https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/conditional_flow_matching.py

import torch
from torch import Tensor


class FlowMatcher:
    def __init__(self, t_schedule) -> None:
        self.t_schedule = t_schedule

    def sample(self, x0: Tensor, x1: Tensor) -> list[Tensor, Tensor, Tensor]:
        r"""
        Args:
            x0: (b, ...), noise
            x1: (b, ...), data

        Returns:
            t:  (b,)
            xt: (b, ...), input
            ut: (b, ...), target
        """

        # Randomly sample t between [0, 1]
        t = self.t_schedule.sample(x0.shape[0]).to(x0.device)  # (b,)

        t_ = t.view(-1, *([1] * (x0.ndim - 1)))  # (b, ...)

        # Interpolation
        xt = (1 - t_) * x0 + t_ * x1  # (b, ...)
        ut = x1 - x0  # (b, ...)

        return t, xt, ut