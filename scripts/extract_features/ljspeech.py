import argparse
from pathlib import Path
import pandas as pd

from audioflow.encoders.audio import load_encoder
from audioflow.utils.audio import extract_and_save_audio_features, load_stereo
from audioflow.utils.misc import augment_path
from audioflow.utils.text import write_lines


def extract_audio_features(args) -> None:

    # Arguments
    root = Path(args.dataset_root)
    split = args.split
    encoder_name = args.encoder_name
    chunk_duration = args.chunk_duration
    device = args.device
    out_dir = Path(args.out_dir)

    # Load audio encoder
    encoder = load_encoder(encoder_name).to(device)

    split_path = root / f"{split}.txt"
    split_names = load_names(split_path)
    n_data = len(split_names)

    for n in range(n_data):

        print(f"{n}/{n_data}")
        path = root / "wavs" / f"{split_names[n]}.wav"
        audio = load_stereo(path, encoder.sr)  # (2, l)
        
        chunk_samples = int(chunk_duration * encoder.sr)
        out_path = out_dir / f"{path.stem}.h5"
        
        extract_and_save_audio_features(
            audio=audio, 
            aug_repeats=1, 
            chunk_samples=chunk_samples, 
            model=encoder, 
            encoder_name=encoder_name, 
            out_path=out_path
        )


def extract_texts(args) -> None:

    # Arguments
    root = Path(args.dataset_root)
    split = args.split
    out_dir = Path(args.out_dir)

    split_path = root / f"{split}.txt"
    meta_path = root / "metadata.csv"
    meta_dict = load_meta(split_path, meta_path)
    n_data = len(meta_dict["name"])
    
    for n in range(n_data):

        print(f"{n}/{n_data}")
        name = meta_dict["name"][n]
        text = meta_dict["text"][n]
        
        out_path = out_dir / f"{name}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_lines(out_path, [text])
        print(f"Write out to {out_path}")


def load_names(split_path: str) -> list[str]:
    df = pd.read_csv(split_path, header=None)
    names = df[0].values
    return names


def load_meta(split_path: str, meta_path: str) -> dict:
    split_names = load_names(split_path)

    df = pd.read_csv(meta_path, sep="|", header=None)
    meta_dict = {"name": [], "text": []}

    for n in range(len(df)):
        name = df[0][n]
        text = df[1][n]
        if name in split_names:
            meta_dict["name"].append(name)
            meta_dict["text"].append(text)

    return meta_dict


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

    parser_text = subparsers.add_parser("text")
    parser_text.add_argument("--dataset_root", type=str, required=True)
    parser_text.add_argument("--split", type=str, required=True)
    parser_text.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "audio":
        extract_audio_features(args)
    
    elif args.mode == "text":
        extract_texts(args)

    else:
        raise ValueError