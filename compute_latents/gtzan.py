import argparse
import os
from pathlib import Path

import librosa
import numpy as np

from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.utils import load_vae, load_stereo, compute_and_save_latents


def compute_vae(args) -> None:

    # Arguments
    root = args.dataset_root
    split = args.split
    latent_type = args.latent_type
    aug_repeats = args.augmentation_repeats
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir
    device = "cuda"

    # Load VAE
    vae = load_vae(latent_type).to(device)

    # Compuate VAE latent
    labels = sorted(os.listdir(Path(root, "genres")))

    for k, label in enumerate(labels):
        print("{}/{}".format(k, len(labels)))
        paths = sorted(list(Path(root, "genres", label).glob("*.au")))

        if split == "train":
            paths = paths[0 : 90]
        elif split == "test":
            paths = paths[90 :]
        else:
            raise ValueError(split)
        
        for path in paths:
            audio = load_stereo(path, vae.sr)  # (2, l)
            
            chunk_samples = int(chunk_duration * vae.sr)
            base_path = Path(out_dir, path.stem)
            
            compute_and_save_latents(
                audio=audio, 
                aug_repeats=aug_repeats, 
                chunk_samples=chunk_samples, 
                model=vae, 
                latent_type=latent_type, 
                base_path=base_path
            )

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--latent_type", type=str, default="levo_vae")
    parser.add_argument("--augmentation_repeats", type=int, default=10)
    parser.add_argument("--chunk_duration", type=float, default=60.)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    compute_vae(args)