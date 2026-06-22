from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import soundfile
import torch
from torch import Tensor
import torchdiffeq
from torch.utils.data._utils.collate import default_collate
import numpy as np
import math

from audio_flow.utils import parse_yaml, to_device, load_vae, load_stereo
from audio_flow.solvers.euler import euler_solver
from train import get_model


def sample(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    config_path = args.config
    ckpt_path = args.ckpt_path
    out_path = args.out_path
    duration = args.duration
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]

    # Load model
    model = get_model(configs, ckpt_path).to(device)
    
    # Load VAE
    vae = load_vae("levo_vae").to(device)

    # Prepare meta data
    duration = get_duration(args)

    # Noise
    length = round(duration * vae.fps)
    noise = torch.randn(1, length, vae.dim).to(device)
    
    with torch.no_grad():
        model.eval()
        data = get_data(args, duration, vae.fps)
        data = default_collate([data])
        data = to_device(data, device)

        controls = model.adapter(data)
        x_gen = euler_solver(model.base, noise, controls, n_steps=100)  # (b, l, d)
    
    # Decode audio from VAE latents
    audio_gen = vae.decode(x_gen).data.cpu().numpy()[0]  # (c, l)

    # Write out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(file=out_path, data=audio_gen.T, samplerate=vae.sr)
    print(f"Write out to {out_path}")


def get_data(args, duration, fps) -> dict:

    task = args.task
    configs = parse_yaml(args.config)

    if task in ["text to music", "text to speech", "text to audio"]:
        return {
            "task": task, 
            "prompt": args.prompt,
            "target_mask": np.ones(round(duration * fps), dtype=bool)
        }

    elif task in ["music source separation", "vocals to music", 
        "mono to stereo", "super-resolution", "codec to music"]:
        return {
            "task": task,
            "input_latent": compute_audio_vae(args.input_path, configs, duration),
            "target_mask": np.ones(int(duration * fps), dtype=bool)
        }

    elif task in ["audio editing"]:
        return {
            "task": task,
            "prompt": args.prompt,
            "input_latent": compute_audio_vae(args.input_path, configs, duration),
            "target_mask": np.ones(int(duration * fps), dtype=bool)
        }

    elif task in ["midi to audio"]:
        return {
            "task": task,
            "input_latent": compute_midi_roll(args.input_path, configs, duration),
            "target_mask": np.ones(int(duration * fps), dtype=bool)
        }

    else:
        raise ValueError(task)


def get_duration(args):
    task = args.task

    if task == "text to speech":
        return len(args.prompt) / 16.30

    else:
        return args.duration


def compute_audio_vae(audio_path: str, configs: dict, duration: float) -> Tensor:

    # configs = parse_yaml(args.config)
    device = configs["train"]["device"]
    vae = load_vae("levo_vae").to(device)

    sr = vae.sr
    audio = load_stereo(audio_path, sr)  # (c, l)
    audio = librosa.util.fix_length(data=audio, size=int(duration * sr), axis=-1)
    audio = Tensor(audio).to(device)  # (c, l)
    latent = vae.encode(audio[None, :, :])[0]  # (t, d)
    return latent


def compute_midi_roll(midi_path: str, configs: dict, duration: float) -> Tensor:

    from compute_latents.midi_io import read_single_track_midi

    fps = 100
    notes, pedals = read_single_track_midi(midi_path=midi_path, extend_pedal=True)

    midi_duration = max([note.end for note in notes])
    n_frames = math.ceil(midi_duration * fps)
    frame_roll = np.zeros((n_frames, 128), dtype=bool)
    onset_roll = np.zeros((n_frames, 128), dtype=bool)

    for note in notes:
        start = round(note.start * fps)
        end = round(note.end * fps)
        pitch = note.pitch
        velocity = note.velocity
        frame_roll[start : end, pitch] = True
        onset_roll[start, pitch] = True

    latent = np.concatenate([frame_roll, onset_roll], axis=-1)  # (l, d)
    latent = latent[0 : int(duration) * fps, :]
    return latent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=False)
    parser.add_argument("--out_path", type=str, required=True)

    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--input_path", type=str)
    
    args = parser.parse_args()

    sample(args)