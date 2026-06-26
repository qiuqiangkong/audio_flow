import argparse
from pathlib import Path

# import pandas as pd

from audioflow.encoders.audio import load_encoder
from audioflow.utils.audio import extract_and_save_audio_features, load_stereo
from audioflow.utils.misc import augment_path
from audioflow.utils.text import write_lines
from audioflow.utils.json import read_jsonl


def extract_audio_features(args) -> None:

    # Arguments
    audios_dir = Path(args.audios_dir)
    encoder_name = args.encoder_name
    chunk_duration = args.chunk_duration
    device = args.device
    out_dir = Path(args.out_dir)

    # Load audio encoder
    encoder = load_encoder(encoder_name).to(device)

    paths = sorted(audios_dir.glob("*.flac"))
    n_data = len(paths)

    for n in range(n_data):
        print(f"{n}/{n_data}")
        path = paths[n]
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
    json_path = Path(args.json_path)
    out_dir = Path(args.out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_jsonl(json_path)
    data = data[0]["data"]
    n_data = len(data)

    for n in range(n_data):
        name = Path(data[n]["id"])
        caption = data[n]["caption"]
        out_path = out_dir / f"{name.stem}.txt"
        
        write_lines(out_path, [caption])
        print(f"Write out to {out_path}")

    print(f"Total: {n_data}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_audio = subparsers.add_parser("audio")
    parser_audio.add_argument("--audios_dir", type=str, required=True)
    parser_audio.add_argument("--encoder_name", type=str, required=True)
    parser_audio.add_argument("--chunk_duration", type=float, default=60.)
    parser_audio.add_argument("--device", type=str, default="cuda")
    parser_audio.add_argument("--out_dir", type=str, required=True)

    parser_text = subparsers.add_parser("text")
    parser_text.add_argument("--json_path", type=str, required=True)
    parser_text.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "audio":
        extract_audio_features(args)
    
    elif args.mode == "text":
        extract_texts(args)

    else:
        raise ValueError