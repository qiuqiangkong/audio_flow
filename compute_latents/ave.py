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
from torchvision.io import read_video
from transformers import VideoMAEImageProcessor, VideoMAEModel
from decord import VideoReader
import torch.nn.functional as F

from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.utils import load_vae, load_stereo, extract_latents_in_chunks
from audio_flow.encoders.image.clip import CLIPEncoder



def compute_audio_vae(args) -> None:

    # Arguments
    root = args.dataset_root
    split = args.split
    latent_type = args.latent_type
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir
    device = "cuda"

    # Load VAE
    vae = load_vae(latent_type).to(device)

    csv_path = Path(root, f"{split}Set.txt")
    meta_dict = load_meta(csv_path)
    n_data = len(meta_dict["name"])

    for n in range(n_data):
        name = meta_dict["name"][n]
        print("{}/{}, {}".format(n, n_data, name))

        path = Path(root, "AVE", f"{name}.mp4")
        audio = load_stereo(path, vae.sr)  # (2, l)
        audio = librosa.util.fix_length(data=audio, size=int(10. * vae.sr), axis=-1)
        
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


def compute_video_latent(args) -> None:

    # Arguments
    root = args.dataset_root
    split = args.split
    latent_type = args.latent_type
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir
    device = "cuda"

    # Load VAE
    encoder = CLIPEncoder().to(device)

    csv_path = Path(root, f"{split}Set.txt")
    meta_dict = load_meta(csv_path)
    n_data = len(meta_dict["name"])

    for n in range(n_data):
        name = meta_dict["name"][n]
        print("{}/{}, {}".format(n, n_data, name))

        path = Path(root, "AVE", f"{name}.mp4")
        tmp_path = "_tmp.mp4"
        cmd = f"ffmpeg -y -loglevel panic -i {path} -r 25 {tmp_path}"
        os.system(cmd)

        video, _, info = read_video(tmp_path, output_format="TCHW", pts_unit="sec")
        latent = extract_images_latents_in_chunks(encoder, video.numpy(), chunk_size=64)

        out_path = Path(out_dir, Path(name).stem + ".h5")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("latent", data=latent, dtype=np.float32)
            hf.attrs.create("fps", data=info["video_fps"], dtype=float)
            hf.attrs.create("duration", data=latent.shape[0] / info["video_fps"], dtype=float)
            hf.attrs.create("latent_type", data=latent_type)

        print(f"Write out to {out_path} {latent.shape}")


def compute_video_mae_latent(args) -> None:

    # Arguments
    root = args.dataset_root
    split = args.split
    latent_type = args.latent_type
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir
    device = "cuda"

    # Load VAE
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    encoder.eval()

    csv_path = Path(root, f"{split}Set.txt")
    meta_dict = load_meta(csv_path)
    n_data = len(meta_dict["name"])

    for n in range(n_data):
        name = meta_dict["name"][n]
        print("{}/{}, {}".format(n, n_data, name))

        path = Path(root, "AVE", f"{name}.mp4")
        tmp_path = "_tmp.mp4"
        cmd = f"ffmpeg -y -loglevel panic -i {path} -r 24 {tmp_path}"
        os.system(cmd)

        vr = VideoReader(tmp_path)
        indices = np.arange(0, 192, 12)
        frames = vr.get_batch(indices).asnumpy()
        inputs = processor(list(frames), return_tensors="pt")
        pixel_values = (inputs["pixel_values"].to(device))

        with torch.no_grad():
            outputs = encoder(pixel_values)

        latent = outputs.last_hidden_state

        B = latent.shape[0]
        D = latent.shape[2]
        latent = latent.reshape(B, 8, 196, D)  # (B, 8, 196, 768)
        latent = latent.mean(dim=2)  # (b, t, d)
        latent = F.pad(latent, pad=(0, 0, 0, 2), mode="replicate")
        latent = latent[0].cpu().numpy()

        out_path = Path(out_dir, Path(name).stem + ".h5")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("latent", data=latent, dtype=np.float32)
            hf.attrs.create("fps", data=1., dtype=float)
            hf.attrs.create("duration", data=10., dtype=float)
            hf.attrs.create("latent_type", data=latent_type)

        print(f"Write out to {out_path} {latent.shape}")


def load_meta(meta_csv: str) -> dict:
    
    df = pd.read_csv(meta_csv, sep="&", header=None)

    return {
        "label": df[0].values,
        "name": df[1].values
    }


def extract_images_latents_in_chunks(
    model: nn.Module, 
    images: np.array, 
    chunk_size: int,
) -> np.array:
    r"""Convert audio into latents.
    
    Args:
        model (nn.Module)
        x (np.ndarray): (b, c, h, w)

    Returns:
        out: (d, t)
    """
    device = next(model.parameters()).device
    latents = []
    i = 0
    
    while i < images.shape[0]:
        x = torch.from_numpy(images[i : i + chunk_size, ...]).to(device)  # (b, c, h, w)

        with torch.no_grad():
            model.eval()
            latent = model(x).data.cpu().numpy()  # (d, t)

        latents.append(latent)
        i += chunk_size

    return np.concatenate(latents, axis=0)


def extract_text(args):

    # Arguments
    root = args.dataset_root
    split = args.split
    out_dir = args.out_dir

    csv_path = Path(root, f"{split}Set.txt")
    df = pd.read_csv(csv_path, sep="&", header=None)
    labels = df[0].values
    names = df[1].values

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for label, name in zip(labels, names):
        out_path = Path(out_dir, f"{name}.txt")

        with open(out_path, 'w') as fw:
            fw.write(label)

        print(f"Write out to {out_path}")
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_audio = subparsers.add_parser("audio")
    parser_audio.add_argument("--dataset_root", type=str, required=True)
    parser_audio.add_argument("--split", type=str, required=True)
    parser_audio.add_argument("--latent_type", type=str, default="levo_vae")
    parser_audio.add_argument("--chunk_duration", type=float, default=60.)
    parser_audio.add_argument("--out_dir", type=str, required=True)

    parser_video = subparsers.add_parser("video")
    parser_video.add_argument("--dataset_root", type=str, required=True)
    parser_video.add_argument("--split", type=str, required=True)
    parser_video.add_argument("--latent_type", type=str, default="clip")
    parser_video.add_argument("--chunk_duration", type=float, default=60.)
    parser_video.add_argument("--out_dir", type=str, required=True)

    parser_video = subparsers.add_parser("video_mae")
    parser_video.add_argument("--dataset_root", type=str, required=True)
    parser_video.add_argument("--split", type=str, required=True)
    parser_video.add_argument("--latent_type", type=str, default="clip")
    parser_video.add_argument("--chunk_duration", type=float, default=60.)
    parser_video.add_argument("--out_dir", type=str, required=True)

    parser_video = subparsers.add_parser("text")
    parser_video.add_argument("--dataset_root", type=str, required=True)
    parser_video.add_argument("--split", type=str, required=True)
    parser_video.add_argument("--out_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.mode == "audio":
        compute_audio_vae(args)

    elif args.mode == "video":
        compute_video_latent(args)

    elif args.mode == "video_mae":
        compute_video_mae_latent(args)

    elif args.mode == "text":
        extract_text(args)

    else:
        raise ValueError