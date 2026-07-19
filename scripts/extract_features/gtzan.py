import argparse
from pathlib import Path

from audioflow.encoders.audio import load_encoder
from audioflow.utils.audio import extract_and_save_audio_features, load_stereo
from audioflow.utils.misc import augment_path
from audioflow.utils.text import write_lines


def extract_audio_features(args) -> None:

    # Arguments
    root = Path(args.dataset_root)
    split = args.split
    encoder_name = args.encoder_name
    aug_repeats = args.augmentation_repeats
    chunk_duration = args.chunk_duration
    device = args.device
    out_dir = Path(args.out_dir)

    # Load audio encoder
    encoder = load_encoder(encoder_name).to(device)

    audios_dir = root / "genres"
    meta_dict = load_meta(audios_dir, split)
    n_data = len(meta_dict["path"])
    
    for n in range(n_data):

        print(f"{n}/{n_data}")
        path = meta_dict["path"][n]
        audio = load_stereo(path, encoder.sr)  # (2, l)
        
        chunk_samples = int(chunk_duration * encoder.sr)
        out_path = out_dir / f"{path.stem}.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        extract_and_save_audio_features(
            audio=audio, 
            aug_repeats=aug_repeats, 
            chunk_samples=chunk_samples, 
            model=encoder, 
            encoder_name=encoder_name, 
            out_path=out_path
        )


def extract_texts(args) -> None:

    # Arguments
    root = Path(args.dataset_root)
    split = args.split
    aug_repeats = args.augmentation_repeats
    out_dir = Path(args.out_dir)

    audios_dir = root / "genres"
    meta_dict = load_meta(audios_dir, split)
    n_data = len(meta_dict["path"])
    
    for n in range(n_data):

        print(f"{n}/{n_data}")
        label = meta_dict["label"][n]
        path = meta_dict["path"][n]
        
        out_path = out_dir / f"{path.stem}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for i in range(aug_repeats):
            aug_path = augment_path(out_path, i)
            write_lines(aug_path, [label])
            print(f"Write out to {aug_path}")


def load_meta(audios_dir: Path, split: str) -> dict:
    r"""Get train and test split paths.
    """
    labels = sorted(p.name for p in audios_dir.iterdir())
    meta_dict = {"label": [], "path": []}

    for label in labels:
        class_paths = sorted((audios_dir / label).glob("*.au"))

        if split == "train":
            paths = class_paths[0 : 90]

        elif split == "test":
            paths = class_paths[90 :]

        else:
            raise ValueError(split)

        meta_dict["label"].extend([label] * len(paths))
        meta_dict["path"].extend(paths)

    return meta_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_audio = subparsers.add_parser("audio")
    parser_audio.add_argument("--dataset_root", type=str, required=True)
    parser_audio.add_argument("--split", type=str, required=True)
    parser_audio.add_argument("--encoder_name", type=str, required=True)
    parser_audio.add_argument("--augmentation_repeats", type=int, default=10)
    parser_audio.add_argument("--chunk_duration", type=float, default=60.)
    parser_audio.add_argument("--device", type=str, default="cuda")
    parser_audio.add_argument("--out_dir", type=str, required=True)

    parser_text = subparsers.add_parser("text")
    parser_text.add_argument("--dataset_root", type=str, required=True)
    parser_text.add_argument("--split", type=str, required=True)
    parser_text.add_argument("--augmentation_repeats", type=int, default=10)
    parser_text.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "audio":
        extract_audio_features(args)
    
    elif args.mode == "text":
        extract_texts(args)

    else:
        raise ValueError