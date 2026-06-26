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
    subset = args.subset
    encoder_name = args.encoder_name
    chunk_duration = args.chunk_duration
    device = args.device
    out_dir = Path(args.out_dir)

    # Load audio encoder
    encoder = load_encoder(encoder_name).to(device)

    # Path dict
    paths = list(root.rglob('*.mp3'))
    path_dict = {p.stem: p for p in paths}

    # Meta
    meta_csv = root / "tracks.csv"
    meta_dict = load_meta(meta_csv, split, subset, parquet_path=None)
    n_data = len(meta_dict["name"])

    for n in range(n_data):
        print(f"{n}/{n_data}")

        name = meta_dict["name"][n]
        path = path_dict[name]
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
    subset = args.subset
    out_dir = Path(args.out_dir)

    # Meta
    meta_csv = root / "tracks.csv"
    parquet_path = root / "train-00000-of-00001.parquet"
    meta_dict = load_meta(meta_csv, split, subset, parquet_path)
    n_data = len(meta_dict["name"])
    
    for n in range(n_data):

        print(f"{n}/{n_data}")
        captions = collect_captions([meta_dict["salmonn_text"][n], meta_dict["chatgpt_texts"][n]])
        print(captions)

        path = Path(meta_dict["name"][n])
        out_path = out_dir / f"{path.stem}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        write_lines(out_path, captions)
        print(f"Write out to {out_path}")


def load_meta(csv_path: str, split: str, subset: str, parquet_path=None) -> dict:
    
    split_mapping = {
        "train": "training",
        "validation": "validation",
        "test": "test"
    }

    df = pd.read_csv(csv_path, sep=",", header=1)
    track_ids = df["Unnamed: 0"].values[1:]
    splits = df["split"].values[1:]
    subsets = df["subset"].values[1:]
    del df

    indices = (splits == split_mapping[split]) & (subsets == subset)
    names = [str(e).zfill(6) for e in track_ids[indices]]

    if parquet_path is None:
        meta_dict = {
            "name": names,
        }
        return meta_dict

    else:
        df = pd.read_parquet(parquet_path)
        df = df.set_index("id")
        meta_dict = {"name": [], "salmonn_text": [], "chatgpt_texts": []}

        for name in names:
            if name in df.index:
                meta_dict["name"].append(name)
                meta_dict["salmonn_text"].append(df.loc[name]["salmonn_text"])
                meta_dict["chatgpt_texts"].append(df.loc[name]["chatgpt_texts"][0])

    return meta_dict


def collect_captions(captions):
    captions = [cap for cap in captions if cap]
    return captions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_audio = subparsers.add_parser("audio")
    parser_audio.add_argument("--dataset_root", type=str, required=True)
    parser_audio.add_argument("--split", type=str, required=True)
    parser_audio.add_argument("--subset", type=str, required=True)
    parser_audio.add_argument("--encoder_name", type=str, required=True)
    parser_audio.add_argument("--chunk_duration", type=float, default=60.)
    parser_audio.add_argument("--device", type=str, default="cuda")
    parser_audio.add_argument("--out_dir", type=str, required=True)

    parser_text = subparsers.add_parser("text")
    parser_text.add_argument("--dataset_root", type=str, required=True)
    parser_text.add_argument("--split", type=str, required=True)
    parser_text.add_argument("--subset", type=str, required=True)
    parser_text.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "audio":
        extract_audio_features(args)
    
    elif args.mode == "text":
        extract_texts(args)

    else:
        raise ValueError