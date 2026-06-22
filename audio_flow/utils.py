from __future__ import annotations

import json
from pathlib import Path
from torch import Tensor

import h5py
import librosa
import numpy as np
import re
import torch
import torch.nn as nn
import yaml

import random


def load_vae(vae_type: str) -> nn.Module:
    if vae_type == "levo_vae":
        from audio_flow.encoders.audio.levo_vae import LevoVAE
        return LevoVAE()

    elif vae_type == "architts_vae":
        from audio_flow.encoders.audio.architts_vae import ArchiTTSVAE
        return ArchiTTSVAE()

    elif vae_type == "mmaudio_vae":
        from audio_flow.encoders.audio.mmaudio_vae import MMAudioVAE
        return MMAudioVAE()

    else:
        raise ValueError(vae_type)


def load_stereo(path: str, sr: int) -> np.ndarray:
    audio, fs = librosa.load(path=path, sr=sr, mono=False)

    if audio.ndim == 1:
        return np.repeat(audio[None, :], repeats=2, axis=0)  # (2, l)
    
    elif audio.ndim == 2:
        if audio.shape[0] == 1:
            return np.repeat(audio[None, :], repeats=2, axis=0)  # (2, l)

        elif audio.shape[0] == 2:
            return audio
            
        else:
            return np.repeat(np.mean(audio, axis=0, keepdims=True), repeats=2, axis=0)  # (2, l)



def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


def load_jsonl(path) -> list[dict]:
    with open(path, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    return lines


def write_jsonl(metas: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for meta in metas:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def get_single_value(lst: list):
    unique = list(set(lst))
    assert len(unique) == 1
    return unique[0]


def truncate_latent(data: dict) -> dict:
    for key in ["target_latent", "target_mask"]:
        data[key] = data[key][:, 0 : max(data["target_length"])]
    return data


def to_device(data: dict, device) -> dict:
    for key, value in data.items():
        if isinstance(value, Tensor):
            data[key] = value.to(device)
    return data



class LinearWarmUp:
    r"""Linear learning rate warm up scheduler."""

    def __init__(self, warm_up_steps: int) -> None:
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        if step <= self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            return 1.


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float = 0.999) -> None:
    """Update EMA model weights and buffers from model."""

    # Parameters
    for e, m in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(m.data.float(), alpha=1 - decay)

    # Buffers (BN running stats, etc)
    for e, m in zip(ema.buffers(), model.buffers()):
        if m.dtype in [torch.bool, torch.long]:
            continue
        e.mul_(decay).add_(m.data.float(), alpha=1 - decay)


def requires_grad(model: nn.Module, flag=True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


class CombinedModel(nn.Module):
    def __init__(self, base: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter


def extract_latents_in_chunks(
    model: nn.Module, 
    audio: np.array, 
    chunk_samples: int,
    min_tail_samples: int = 10000
) -> np.array:
    r"""Convert audio into latents.

    c: audio_channels
    l: audio_samples
    d: dim
    t: time_steps

    Args:
        model (nn.Module)
        audio (np.ndarray): (c, l)
        chunk_samples (int)
        min_tail_samples (int)

    Returns:
        out: (d, t)
    """
    device = next(model.parameters()).device
    latents = []
    total_samples = audio.shape[-1]
    i = 0
    
    while i < total_samples:
        remaining_samples = total_samples - i
        if remaining_samples < min_tail_samples:
            break
        
        x = torch.from_numpy(audio[None, :, i : i + chunk_samples]).to(device)  # (b, c, l)

        with torch.no_grad():
            model.eval()
            latent = model(x)[0].data.cpu().numpy()  # (d, t)

        latents.append(latent)
        i += chunk_samples

    return np.concatenate(latents, axis=0)


def compute_and_save_latents(
    audio: np.ndarray, 
    aug_repeats: int, 
    chunk_samples: int, 
    model: nn.Module, 
    latent_type: str,
    base_path: str,
    dtype=np.float32
) -> None:
    r"""Convert audio into latents and write to HDF5.

    c: audio_channels
    l: audio_samples
    d: dim
    t: time_steps

    Args:
        audio (np.ndarray): (c, l)
        aug_repeats (int), number of jitter repetitions for data augmentation

    Returns:
        None
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(aug_repeats):
                
        jitter = round((i / aug_repeats) * (model.sr / model.fps))
        x = audio[:, jitter :]  # (2, l)
        latent = extract_latents_in_chunks(model, x, chunk_samples)  # (t, d)

        if aug_repeats == 1:
            out_path = str(base_path) + ".h5"
        else:
            out_path = str(base_path) + f"_{i:03d}_of_{aug_repeats:03d}.h5"
            
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("latent", data=latent, dtype=dtype)
            hf.attrs.create("fps", data=model.fps, dtype=float)
            hf.attrs.create("duration", data=x.shape[-1] / model.sr, dtype=float)
            hf.attrs.create("latent_type", data=latent_type)

        print(f"Write out to {out_path} {latent.shape}")


def logmel(audio: np.ndarray, sr: float) -> np.ndarray:

    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)

    return np.log10(librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=2048, 
        hop_length=round(sr * 0.01), 
        n_mels=128
    )).T  # (t, f)




def build_attention_mask(mask):
    r"""Build mask."""
    return mask[:, None, :, None] * mask[:, None, None, :]  # (b, 1, l, l)





# def normalize_text(x: str) -> str:
#     from IPython import embed; embed(using=False); os._exit(0)
#     x = re.sub(r"[^\w\s]", " ", x.lower())  # Remain char and digit only
#     x = re.sub(r"\s+", " ", x)  # Remove extra spaces
#     return x.strip()



def mean_pool(x: Tensor, mask: Tensor, keepdims=False) -> Tensor:
    r"""

    Args:
        x: (b, l, d)
        mask: (b, l)

    Returns:
        out: (b, d)
    """
    out = (x * mask[:, :, None]).sum(1) / mask.sum(dim=1, keepdims=True)
    
    if keepdims:
        out = out[:, None, :]

    return out


def get_saveable_state_dict(model: nn.Module) -> dict:
    
    excluded = [n for n, m in model.named_modules() if hasattr(m, "saveable") and m.saveable is False]

    save_dict = {
        k: v for k, v in model.state_dict().items()
        if not any(k == p or k.startswith(p + ".") for p in excluded)
    }

    return save_dict


def get_audio_latent_length(path: str) -> int:
    with h5py.File(path, 'r') as hf:
        return hf["latent"].shape[0]


def sample_start_frame(total_frames: int, clip_frames: int) -> int:
    r"""Random sample a frame index."""
    max_start = max(total_frames - clip_frames, 0)
    return random.randint(0, max_start)


def load_audio_latent(path: str, start: int, clip_frames: int) -> tuple[np.ndarray, np.ndarray, int]:
    r"""Load latent from hdf5."""
    with h5py.File(path, 'r') as hf:
        latent = hf["latent"][start : start + clip_frames, :]  # (l, d)
        length = latent.shape[0]

        latent = librosa.util.fix_length(data=latent, size=clip_frames, axis=0, constant_values=0.)
        mask = np.zeros(clip_frames, dtype=bool)
        mask[:length] = True

    return latent, mask, length


def check_masks_type(masks: list[Tensor], dtype) -> bool:
    return all(mask.dtype == dtype for mask in masks)