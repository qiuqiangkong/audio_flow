from __future__ import annotations

import os
import sys
from collections import OrderedDict
from contextlib import contextmanager

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor


def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


class LinearWarmUp:
    r"""Linear learning rate warm up scheduler.
    """
    def __init__(self, warm_up_steps: int) -> None:
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        if step <= self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            return 1.


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay=0.999) -> None:

    # Moving average of parameters
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # Moving average of buffers. Patch for BN, etc
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        if buffer.dtype in [torch.long]:
            continue
        ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)


def requires_grad(model: nn.Module, flag=True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def fix_length(x: Tensor, size: int) -> Tensor:

    if x.shape[-1] >= size:
        return x[:, :, 0 : size]
    else:
        pad_t = size - x.shape[-1]
        return F.pad(input=x, pad=(0, pad_t))


@contextmanager
def suppress_print():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


class Logmel:
    def __init__(self, sr: float):
        self.sr = sr
        self.n_fft = 2048
        self.hop_length = round(sr * 0.01)
        self.n_mels = 128

    def __call__(self, audio: np.array) -> np.array:
        
        logmel = np.log10(librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )).T

        return logmel
