import argparse
from pathlib import Path
import numpy as np
import librosa

from audioflow.encoders.audio import load_encoder
from audioflow.utils.audio import extract_and_save_audio_features, load_stereo
from audioflow.utils.misc import augment_path
from audioflow.utils.text import write_lines


def extract_audio_features(args) -> None:

    # Arguments
    root = Path(args.dataset_root)
    split = args.split
    stem = args.stem
    degradation = args.degradation
    encoder_name = args.encoder_name
    aug_repeats = args.augmentation_repeats
    chunk_duration = args.chunk_duration
    device = args.device
    out_dir = Path(args.out_dir)

    # Load audio encoder
    encoder = load_encoder(encoder_name).to(device)

    audio_paths = sorted((root / split).glob("*"))
    n_data = len(audio_paths)

    for n in range(n_data):

        print(f"{n}/{n_data}")
        path = Path(audio_paths[n], f"{stem}.wav")
        audio = load_stereo(path, encoder.sr)  # (2, l)

        if degradation != "":
            audio = degrade(audio, degradation, encoder.sr)  # (2, l)
        
        chunk_samples = int(chunk_duration * encoder.sr)
        out_path = out_dir / f"{path.parent.stem}.h5"
        
        extract_and_save_audio_features(
            audio=audio, 
            aug_repeats=aug_repeats, 
            chunk_samples=chunk_samples, 
            model=encoder, 
            encoder_name=encoder_name, 
            out_path=out_path
        )


def degrade(audio: np.ndarray, degradation: str, sr: float) -> np.ndarray:
    
    if degradation == "mono":
        return dual_mono(audio)

    elif degradation == "low_res":
        return low_res(audio, sr)

    else:
        raise ValueError(degradation)



def dual_mono(audio: np.ndarray) -> np.ndarray:
    audio = audio.mean(axis=0)
    return np.stack([audio, audio], axis=0)


def low_res(audio: np.ndarray, sr: float) -> np.ndarray:
    low_sr = 8000
    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=low_sr)
    audio = librosa.resample(y=audio, orig_sr=low_sr, target_sr=sr)
    return audio


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_audio = subparsers.add_parser("audio")
    parser_audio.add_argument("--dataset_root", type=str, required=True)
    parser_audio.add_argument("--split", type=str, required=True)
    parser_audio.add_argument("--stem", type=str, required=True)
    parser_audio.add_argument("--degradation", type=str, required=True)
    parser_audio.add_argument("--encoder_name", type=str, required=True)
    parser_audio.add_argument("--augmentation_repeats", type=int, default=10)
    parser_audio.add_argument("--chunk_duration", type=float, default=60.)
    parser_audio.add_argument("--device", type=str, default="cuda")
    parser_audio.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "audio":
        extract_audio_features(args)

    else:
        raise ValueError