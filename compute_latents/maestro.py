import argparse
import os
import random
from pathlib import Path

import h5py
import math
import librosa
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pretty_midi import ControlChange, Note, PrettyMIDI

from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.utils import load_vae, load_stereo, extract_latents_in_chunks
from .midi_io import read_single_track_midi


def compute_vae(args) -> None:

    # Arguments
    root = args.dataset_root
    split = args.split
    latent_type = args.latent_type
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir
    device = "cuda"

    # Load VAE
    vae = load_vae(latent_type).to(device)

    csv_path = Path(root, "maestro-v3.0.0.csv")
    meta_dict = load_meta(csv_path, split)
    n_audios = len(meta_dict["audio_name"])

    for n in range(n_audios):
        name = meta_dict["audio_name"][n]
        print("{}/{}, {}".format(n, n_audios, name))
        
        path = Path(root, name)
        audio = load_stereo(path, vae.sr)  # (2, l)

        chunk_samples = int(chunk_duration * vae.sr)
        latent = extract_latents_in_chunks(vae, audio, chunk_samples)  # (t, d)

        out_path = Path(out_dir, Path(name).stem + ".h5")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("latent", data=latent, dtype=np.float32)
            hf.attrs.create("fps", data=vae.fps, dtype=float)
            hf.attrs.create("duration", data=audio.shape[-1] / vae.sr, dtype=float)
            hf.attrs.create("latent_type", data=latent_type)

        print(f"Write out to {out_path} {latent.shape}")


def compute_midi(args) -> None:

    # Arguments
    root = args.dataset_root
    split = args.split
    out_dir = args.out_dir
    device = "cuda"
    fps = 100

    csv_path = Path(root, "maestro-v3.0.0.csv")
    meta_dict = load_meta(csv_path, split)
    n_audios = len(meta_dict["audio_name"])

    for n in range(n_audios):
        name = meta_dict["midi_name"][n]
        print("{}/{}, {}".format(n, n_audios, name))
        
        path = Path(root, name)
        notes, pedals = read_single_track_midi(midi_path=path, extend_pedal=True)
        
        audio_duration = meta_dict["duration"][n]
        midi_duration = max([note.end for note in notes])
        duration = max(audio_duration, midi_duration)
        n_frames = math.ceil(duration * fps)

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

        out_path = Path(out_dir, Path(name).stem + ".h5")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("latent", data=latent, dtype=bool)
            hf.attrs.create("fps", data=fps, dtype=float)
            hf.attrs.create("duration", data=duration, dtype=float)
            hf.attrs.create("latent_type", data="roll")

        print(f"Write out to {out_path} {latent.shape}")

def load_meta(meta_csv: str, split: str) -> dict:
    r"""Load meta dict."""

    df = pd.read_csv(meta_csv, sep=',')
    indexes = df["split"].values == split

    audio_names = df["audio_filename"].values[indexes]
    midi_names = df["midi_filename"].values[indexes]
    durations = df["duration"].values[indexes]

    meta_dict = {
        "audio_name": audio_names,
        "midi_name": midi_names,
        "duration": durations
    }

    return meta_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_audio = subparsers.add_parser("audio")
    parser_audio.add_argument("--dataset_root", type=str, required=True)
    parser_audio.add_argument("--split", type=str, required=True)
    parser_audio.add_argument("--latent_type", type=str, default="levo_vae")
    parser_audio.add_argument("--chunk_duration", type=float, default=60.)
    parser_audio.add_argument("--out_dir", type=str, required=True)

    parser_midi = subparsers.add_parser("midi")
    parser_midi.add_argument("--dataset_root", type=str, required=True)
    parser_midi.add_argument("--split", type=str, required=True)
    parser_midi.add_argument("--out_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.mode == "audio":
        compute_vae(args)

    elif args.mode == "midi":
        compute_midi(args)

    else:
        raise ValueError