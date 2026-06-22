import argparse
from pathlib import Path

import librosa
import numpy as np

from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.utils import load_vae, load_stereo, compute_and_save_latents


def compute_vae(args) -> None:

    # Arguments
    audios_dir = args.audios_dir
    latent_type = args.latent_type
    aug_repeats = args.augmentation_repeats
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir
    device = "cuda"

    # Load VAE
    vae = load_vae(latent_type).to(device)

    paths = list(Path(audios_dir).glob("*.wav"))
    
    for n, path in enumerate(paths):

        print(f"{n}/{len(paths)}")
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
    parser.add_argument("--audios_dir", type=str, required=True)
    parser.add_argument("--latent_type", type=str, default="levo_vae")
    parser.add_argument("--augmentation_repeats", type=int, default=1)
    parser.add_argument("--chunk_duration", type=float, default=60.)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    compute_vae(args)


'''
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

    paths = list(Path(root, "audios", "balanced_train_segments").glob("*.wav"))

    for n, path in enumerate(paths):

        print(f"{n}/{len(paths)}")
        audio = load_stereo(path, vae.sr)  # (2, l)

        chunk_samples = int(chunk_duration * vae.sr)
        base_path = Path(out_dir, "balanced_train_segments", path.stem)
        
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
    parser.add_argument("--audios_dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--latent_type", type=str, default="levo_vae")
    parser.add_argument("--augmentation_repeats", type=int, default=1)
    parser.add_argument("--chunk_duration", type=float, default=60.)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    compute_vae(args)
'''