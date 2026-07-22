from __future__ import annotations

import argparse
from pathlib import Path

from torch import Tensor
import soundfile
import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np
import os
from torchvision.io import read_video

from audioflow.utils.yaml import read_yaml
from audioflow.solvers import get_solver
from audioflow.inference.inference import sample_latent
from audioflow.models import get_model
from audioflow.utils.xml import check_xml

from audioflow.utils.audio import load_stereo, extract_features_in_chunks


def sample(args) -> None:
    r"""Sample."""

    # Arguments
    config_path = args.config
    ckpt_path = Path(args.ckpt_path)
    text = args.text
    in_path = args.input
    out_path = Path(args.out_path)
    duration = args.duration
    device = "cuda"
    check_xml(text)
    
    # Configs
    configs = read_yaml(config_path)
    
    encoder = get_encoder(configs["non_text_condition"]["encoder"], device)
    decoder = get_decoder(configs["output"]["decoder"], device)

    # Load model
    model = get_model(configs["model"], ckpt_path).to(device)
    solver = get_solver(configs["sampling"]["solver"])
    cfg_scale = configs["sampling"]["cfg"]["scale"]

    # Load input
    x = get_input(configs["non_text_condition"]["encoder"]["modality"], in_path, encoder)

    # Split long condition into chunks
    length = round(duration * decoder.fps)
    xs = to_chunks(x, length)

    # Sample every chunck
    outs = []
    for n, x in enumerate(xs):
        print(f"{n}/{len(xs)}")

        data = {}
        data["text"] = [text]
        if x is not None:
            data["input_feature"] = Tensor([x]).to(device)  # (b, l, d)
            data["input_mask"] = torch.ones(1, x.shape[0], dtype=bool).to(device)  # (b, l)
            data["input_fps"] = Tensor([encoder.fps]).to(device)  # (b,)
        
        data["target_mask"] = torch.ones(1, length, dtype=bool).to(device)  # (b, l)
        data["target_fps"] = Tensor([decoder.fps]).to(device)  # (b,)

        # Sample
        dim = configs["non_text_condition"]["encoder"]["dim"]
        noise = torch.randn(1, length, dim).to(device)  # (b, l, d)
        x_gen = sample_latent(model, noise, data, solver, cfg_scale)  # (b, l, d)
        
        # Decode audio from VAE latents
        out = decoder.decode(x_gen).data.cpu().numpy()[0]  # (c, l)
        outs.append(out)

    out = np.concatenate(outs, axis=-1)  # (c, l)
    
    # Write out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(file=out_path, data=out.T, samplerate=decoder.sr)
    print(f"Write out to {out_path}")


def get_encoder(configs: dict, device: str) -> nn.Module | None:
    if configs is None:
        return None

    elif configs["modality"] in ["audio"]:
        from audioflow.encoders.audio import load_encoder
        return load_encoder(configs["name"]).to(device)

    elif configs["modality"] in ["midi"]:
        from audioflow.encoders.midi import load_encoder
        return load_encoder(configs["name"]).to(device)

    elif configs["modality"] in ["video"]:
        from audioflow.encoders.image import load_encoder as load_encoder
        return load_encoder(configs["name"]).to(device)

    else:
        raise NotImplementedError


def get_decoder(configs: dict, device: str) -> nn.Module | None:
    if configs["modality"] in ["audio"]:
        from audioflow.decoders.audio import load_decoder
        return load_decoder(configs["name"]).to(device)

    else:
        raise NotImplementedError


def get_input(modality: str, path: str, encoder: nn.Module | None):
    if modality is None:
        return None

    elif modality in ["midi"]:
        notes, pedals = read_single_track_midi(path, extend_pedal=True)
        return notes

    elif modality in ["audio"]:
        audio = load_stereo(path, encoder.sr)  # (2, l)
        return extract_features_in_chunks(encoder, audio, int(60 * encoder.sr))

    elif modality in ["video"]:
        tmp_path = "__tmp_sample.mp4"
        fps = encoder.fps
        cmd = f"ffmpeg -y -loglevel panic -i {path} -r {fps} {tmp_path}"
        os.system(cmd)
        print("ffmpeg done.")
        video, _, info = read_video(tmp_path, output_format="TCHW", pts_unit="sec")
        return video

    else:
        raise NotImplementedError


def to_chunks(x: np.ndarray, L: int) -> list[np.ndarray]:
    if x is None:
        return [None]
    else:
        return [x[i : i + L, :] for i in range(0, x.shape[0], L)]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--input", type=str)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--duration", type=float, default=10.)
    
    args = parser.parse_args()

    sample(args)