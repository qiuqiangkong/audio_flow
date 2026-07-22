import torch.nn as nn
from torch import Tensor


class AudioFlow(nn.Module):
    def __init__(
        self, 
        adapter: nn.Module,
        backbone: nn.Module, 
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone
        
    def forward(self, t: Tensor, x: Tensor, data: dict) -> Tensor:
        controls = self.adapter(data)
        x = self.backbone(t, x, controls)
        return x


'''
import time
class AudioFlow(nn.Module):
    def __init__(
        self, 
        in_: nn.Module, 
        base: nn.Module, 
        out: nn.Module, 
        adapter: nn.Module
    ) -> None:
        super().__init__()
        self.in_ = in_
        self.out = out
        self.base = base
        self.adapter = adapter

    def forward(self, t: Tensor, x: Tensor, data: dict) -> Tensor:
        t1 = time.time()
        controls = self.adapter(data)
        print("t1:", time.time() - t1)
        t1 = time.time()
        x = self.in_(x)
        x = self.base(t, x, controls)
        x = self.out(x)
        print("t2:", time.time() - t1)
        return x
'''