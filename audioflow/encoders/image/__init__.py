import torch.nn as nn


def load_encoder(name: str) -> nn.Module:
    
    if name == "clip":
        from .clip import CLIPImageEncoder
        return CLIPImageEncoder()

    else:
        raise ValueError(name)
