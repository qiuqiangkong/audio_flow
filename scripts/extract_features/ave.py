import argparse
import os
from pathlib import Path

import pandas as pd
from torchvision.io import read_video

from audioflow.encoders.audio import load_encoder as load_audio_encoder
from audioflow.encoders.image import load_encoder as load_image_encoder
from audioflow.utils.audio import extract_and_save_audio_features, load_stereo
from audioflow.utils.text import write_lines
from audioflow.utils.video import extract_and_save_video_features


def extract_audio_features(args) -> None:

    # Arguments
    root = Path(args.dataset_root)
    split = args.split
    encoder_name = args.encoder_name
    chunk_duration = args.chunk_duration
    device = args.device
    out_dir = Path(args.out_dir)

    # Load audio encoder
    encoder = load_audio_encoder(encoder_name).to(device)

    csv_path = root / f"{split}Set.txt"
    meta_dict = load_meta(csv_path)
    n_data = len(meta_dict["name"])
    
    for n in range(n_data):

        print(f"{n}/{n_data}")
        name = meta_dict["name"][n]
        path = root / "AVE" / (name + ".mp4")
        audio = load_stereo(path, encoder.sr)  # (2, l)
        
        chunk_samples = int(chunk_duration * encoder.sr)
        out_path = out_dir / (name + ".h5")
        
        extract_and_save_audio_features(
            audio=audio, 
            aug_repeats=1, 
            chunk_samples=chunk_samples, 
            model=encoder, 
            encoder_name=encoder_name, 
            out_path=out_path
        )


def extract_video_features(args) -> None:

    # Arguments
    root = Path(args.dataset_root)
    split = args.split
    encoder_name = args.encoder_name
    device = args.device
    out_dir = Path(args.out_dir)
    fps = 25

    # Load video encoder
    encoder = load_image_encoder(encoder_name).to(device)

    csv_path = root / f"{split}Set.txt"
    meta_dict = load_meta(csv_path)
    n_data = len(meta_dict["name"])

    for n in range(n_data):
        print(f"{n}/{n_data}")
        name = meta_dict["name"][n]
        path = root / "AVE" / (name + ".mp4")

        tmp_path = "__tmp.mp4"
        cmd = f"ffmpeg -y -loglevel panic -i {path} -r {fps} {tmp_path}"
        os.system(cmd)
        print("ffmpeg done.")

        video, _, info = read_video(tmp_path, output_format="TCHW", pts_unit="sec")
        out_path = out_dir / (name + ".h5")

        extract_and_save_video_features(
            video=video.numpy(), 
            chunk_frames=64, 
            model=encoder, 
            encoder_name=encoder_name, 
            fps=info["video_fps"],
            out_path=out_path
        )


def extract_texts(args) -> None:

    # Arguments
    root = Path(args.dataset_root)
    split = args.split
    out_dir = Path(args.out_dir)

    csv_path = root / f"{split}Set.txt"
    meta_dict = load_meta(csv_path)
    n_data = len(meta_dict["name"])
    
    for n in range(n_data):
        print(f"{n}/{n_data}")
        label = meta_dict["label"][n]
        name = meta_dict["name"][n]
        
        out_path = out_dir / f"{name}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_lines(out_path, [label])
        print(f"Write out to {out_path}")


def load_meta(meta_csv) -> dict:
    df = pd.read_csv(meta_csv, sep="&", header=None)

    return {
        "label": df[0].values,
        "name": df[1].values
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_audio = subparsers.add_parser("audio")
    parser_audio.add_argument("--dataset_root", type=str, required=True)
    parser_audio.add_argument("--split", type=str, required=True)
    parser_audio.add_argument("--encoder_name", type=str, required=True)
    parser_audio.add_argument("--chunk_duration", type=float, default=60.)
    parser_audio.add_argument("--device", type=str, default="cuda")
    parser_audio.add_argument("--out_dir", type=str, required=True)

    parser_video = subparsers.add_parser("video")
    parser_video.add_argument("--dataset_root", type=str, required=True)
    parser_video.add_argument("--split", type=str, required=True)
    parser_video.add_argument("--encoder_name", type=str, required=True)
    parser_video.add_argument("--device", type=str, default="cuda")
    parser_video.add_argument("--out_dir", type=str, required=True)

    parser_text = subparsers.add_parser("text")
    parser_text.add_argument("--dataset_root", type=str, required=True)
    parser_text.add_argument("--split", type=str, required=True)
    parser_text.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "audio":
        extract_audio_features(args)
    
    elif args.mode == "video":
        extract_video_features(args)

    elif args.mode == "text":
        extract_texts(args)

    else:
        raise ValueError