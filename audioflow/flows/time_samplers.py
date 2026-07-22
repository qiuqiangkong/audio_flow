import torch
from torch import Tensor


class Uniform:
    def sample(self, batch_size: int) -> Tensor:
        return torch.rand(batch_size)


class LogitNormal:
    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def sample(self, batch_size: int) -> Tensor:
        z = self.mu + self.sigma * torch.randn(batch_size)
        t = torch.sigmoid(z)
        return t