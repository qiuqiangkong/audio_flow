from __future__ import annotations

import argparse
from pathlib import Path

import soundfile
import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np

from audioflow.decoders.audio import load_decoder
from audioflow.utils.yaml import read_yaml
from audioflow.utils.misc import check_prompt
from audioflow.solvers import get_solver
from audioflow.inference.sampler import sample_latent
from audioflow.models import get_model


def sample(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    config_path = args.config
    ckpt_path = Path(args.ckpt_path)
    prompt = args.prompt
    out_path = Path(args.out_path)
    duration = args.duration
    device = "cuda"
    
    # Configs
    configs = read_yaml(config_path)
    
    # Load model
    model = get_model(configs["model"], ckpt_path).to(device)
    
    # Load decoder
    vae = load_decoder(configs["vae"]["name"]).to(device)

    # Cfg
    cfg_scale = configs["cfg"]["sample"]["scale"] if "cfg" in configs else None

    # Solver
    solver = get_solver(configs["solver"])

    # Noise
    length = round(duration * vae.fps)
    noise = torch.randn(1, length, vae.dim).to(device)  # (b, l, d)
    
    assert check_prompt(prompt), f"Format error! {prompt}"
    
    data = get_data(prompt, length)
    data = default_collate([data])

    x_gen = sample_latent(model, noise, data, solver, cfg_scale)  # (b, l, d)
    
    # Decode audio from VAE latents
    audio_gen = vae.decode(x_gen).data.cpu().numpy()[0]  # (c, l)

    # Write out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(file=out_path, data=audio_gen.T, samplerate=vae.sr)
    print(f"Write out to {out_path}")


def get_data(prompt: str, length: int) -> dict:
    return {
        "prompt": prompt,
        "target_mask": np.ones(length, dtype=bool)
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)

    parser.add_argument("--duration", type=float, default=10.)
    
    args = parser.parse_args()

    sample(args)