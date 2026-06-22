import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.utils import load_vae, load_stereo, compute_and_save_latents


def compute_vae(args) -> None:

    # Arguments
    root = args.dataset_root
    metadata_root = args.metadata_root
    split = args.split
    subset = args.subset
    latent_type = args.latent_type
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir
    device = "cuda"

    # Load VAE
    vae = load_vae(latent_type).to(device)

    meta_csv = Path(metadata_root, "tracks.csv")
    meta_dict = load_meta(meta_csv)

    split_mapping = {
        "train": "training",
        "validation": "validation",
        "test": "test"
    }

    paths = list(Path(root).rglob('*.mp3'))

    for n, path in enumerate(paths):
        meta = meta_dict[path.stem]

        if meta["split"] == split_mapping[split] and meta["subset"] == subset:
            print(f"{n}/{len(paths)}")
            audio = load_stereo(path, vae.sr)  # (2, l)
            chunk_samples = int(chunk_duration * vae.sr)
            base_path = Path(out_dir, path.stem)
            
            compute_and_save_latents(
                audio=audio, 
                aug_repeats=1, 
                chunk_samples=chunk_samples, 
                model=vae, 
                latent_type=latent_type, 
                base_path=base_path
            )
    

def load_meta(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, sep=",", header=1)
    track_ids = df["Unnamed: 0"].values[1:]
    splits = df["split"].values[1:]
    subsets = df["subset"].values[1:]

    meta_dict = {}
    for i in range(len(track_ids)):
        meta_dict[str(track_ids[i]).zfill(6)] = {
            "split": splits[i],
            "subset": subsets[i]
        }

    return meta_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_audio = subparsers.add_parser("audio")
    parser_audio.add_argument("--dataset_root", type=str, required=True)
    parser_audio.add_argument("--metadata_root", type=str, required=True)
    parser_audio.add_argument("--split", type=str, required=True)
    parser_audio.add_argument("--subset", type=str, required=True)
    parser_audio.add_argument("--latent_type", type=str, default="levo_vae")
    parser_audio.add_argument("--chunk_duration", type=float, default=60.)
    parser_audio.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()
    
    if args.mode == "audio":
        compute_vae(args)

    elif args.mode == "caption":
        compute_video_vae(args)

    else:
        raise ValueError