import torch.nn as nn

from .audioflow import AudioFlow
from audioflow.adapters import get_adapter
from audioflow.utils.torch import load


def get_model(configs: dict, ckpt_path: str) -> nn.Module:

    in_ = get_in(configs["in"])
    out = get_out(configs["out"])
    base = get_base(configs["base"])
    adapter = get_adapter(configs["adapter"])

    model = AudioFlow(in_, base, out, adapter)

    if ckpt_path:
        model = load(model, ckpt_path)
        print(f"Load checkpoint from {ckpt_path}")

    return model


def get_in(configs: dict) -> nn.Module:
    return nn.Linear(configs["in_dim"], configs["dim"])


def get_out(configs: dict) -> nn.Module:
    return nn.Linear(configs["dim"], configs["out_dim"])


def get_base(configs: dict) -> nn.Module:
    r"""Initialize base."""
    name = configs["name"]

    if name == "Transformer":
        from .transformer import Transformer
        return Transformer(**configs)

    else:
        raise ValueError(name)