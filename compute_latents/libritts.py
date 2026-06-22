import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.utils import load_vae, load_stereo, compute_and_save_latents


def compute_vae(args) -> None:

    # Arguments
    audios_dir = args.audios_dir
    latent_type = args.latent_type
    aug_repeats = args.augmentation_repeats
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir
    csv_path = args.csv_path
    device = "cuda"

    # Load VAE
    vae = load_vae(latent_type).to(device)

    audio_paths = list(Path(audios_dir).rglob("*.wav"))
    meta_data = []

    for n, audio_path in enumerate(audio_paths):
        print(f"{n}/{len(audio_paths)}")
        text_path = audio_path.with_suffix(".normalized.txt")

        try:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()

            audio = load_stereo(audio_path, vae.sr)  # (2, l)

            chunk_samples = int(chunk_duration * vae.sr)
            base_path = Path(out_dir, audio_path.stem)
            
            compute_and_save_latents(
                audio=audio, 
                aug_repeats=aug_repeats, 
                chunk_samples=chunk_samples, 
                model=vae, 
                latent_type=latent_type, 
                base_path=base_path
            )

            meta_data.append({
                "name": audio_path.name, 
                "text": text
            })
        except:
            pass

    # Write csv
    # from IPython import embed; embed(using=False); os._exit(0)
    df = pd.DataFrame(meta_data)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, sep='\t', index=False)
    print(f"Write out to {csv_path}")


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--audios_dir", type=str, required=True)
    parser.add_argument("--latent_type", type=str, default="levo_vae")
    parser.add_argument("--augmentation_repeats", type=int, default=1)
    parser.add_argument("--chunk_duration", type=float, default=60.)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    args = parser.parse_args()

    compute_vae(args)