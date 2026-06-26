import argparse
from pathlib import Path

import pandas as pd
import re

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

    csv_path = root / f"LP_MTT_{split}.csv"
    meta_dict = load_meta(csv_path)
    n_data = len(meta_dict["rel_path"])

    print(n_data)
    
    for n in range(n_data):
        
        print(f"{n}/{n_data}")
        rel_path = Path(meta_dict["rel_path"][n])
        path = root / "Audio" / rel_path

        if not path.is_file():
            continue
        
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

    csv_path = root / f"LP_MTT_{split}.csv"
    meta_dict = load_meta(csv_path)
    n_data = len(meta_dict["rel_path"])
    
    for n in range(n_data):

        print(f"{n}/{n_data}")
        captions = [
            meta_dict["caption_writing"][n], 
            meta_dict["caption_summary"][n], 
            meta_dict["caption_paraphrase"][n], 
            meta_dict["caption_attribute_prediction"][n]
        ]

        rel_path = Path(meta_dict["rel_path"][n])
        out_path = out_dir / f"{rel_path.stem}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        write_lines(out_path, captions)
        print(f"Write out to {out_path}")


def load_meta(meta_csv) -> dict:

    df = pd.read_csv(meta_csv, sep=',')

    meta_dict = {
        "rel_path": df["path"].values,
        "caption_writing": df["caption_writing"].fillna("").values,
        "caption_summary": df["caption_summary"].fillna("").values,
        "caption_paraphrase": df["caption_paraphrase"].fillna("").values,
        "caption_attribute_prediction": df["caption_attribute_prediction"].fillna("").values,
    }

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