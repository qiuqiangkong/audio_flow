import argparse
from pathlib import Path
import numpy as np
import librosa
import pandas as pd
import math
import h5py

from audioflow.encoders.audio import load_encoder as load_audio_encoder
from audioflow.encoders.midi import load_encoder as load_midi_encoder
from audioflow.utils.audio import extract_and_save_audio_features, load_stereo
from audioflow.utils.text import write_lines
from audioflow.utils.midi import read_single_track_midi


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

    csv_path = root / "maestro-v3.0.0.csv"
    meta_dict = load_meta(csv_path, split)
    n_data = len(meta_dict["audio_name"])

    for n in range(n_data):
        print(f"{n}/{n_data}")
        path = root / meta_dict["audio_name"][n]
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


def load_meta(meta_csv: str, split: str) -> dict:
    r"""Load meta dict."""

    df = pd.read_csv(meta_csv, sep=',')
    indexes = df["split"].values == split

    audio_names = df["audio_filename"].values[indexes]
    midi_names = df["midi_filename"].values[indexes]
    durations = df["duration"].values[indexes]

    meta_dict = {
        "audio_name": audio_names,
        "midi_name": midi_names,
        "duration": durations
    }

    return meta_dict


def extract_midi(args) -> None:

    # Arguments
    root = Path(args.dataset_root)
    split = args.split
    aug_repeats = args.augmentation_repeats
    out_dir = Path(args.out_dir)
    encoder_name = "frame_onset"
    
    # Load audio encoder
    encoder = load_midi_encoder(encoder_name)

    csv_path = root / "maestro-v3.0.0.csv"
    meta_dict = load_meta(csv_path, split)
    n_data = len(meta_dict["audio_name"])
    
    for n in range(n_data):
        print(f"{n}/{n_data}")
        path = root / meta_dict["midi_name"][n]

        # Read MIDI notes
        notes, pedals = read_single_track_midi(path, extend_pedal=True)
        duration = max([note.end for note in notes])

        # Convert notes to roll
        roll = encoder(notes)  # (t, d)

        out_path = out_dir / f"{path.stem}.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("data", data=roll, dtype=bool)
            hf.attrs.create("fps", data=encoder.fps, dtype=float)
            hf.attrs.create("duration", data=duration, dtype=float)
            hf.attrs.create("type", data=encoder_name)

        print(f"Write out to {out_path} {roll.shape}")



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

    parser_text = subparsers.add_parser("midi")
    parser_text.add_argument("--dataset_root", type=str, required=True)
    parser_text.add_argument("--split", type=str, required=True)
    parser_text.add_argument("--augmentation_repeats", type=int, default=1)
    parser_text.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "audio":
        extract_audio_features(args)
    
    elif args.mode == "midi":
        extract_midi(args)

    else:
        raise ValueError