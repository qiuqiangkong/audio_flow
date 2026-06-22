import argparse
from pathlib import Path

import h5py
import pandas as pd

from audio_flow.utils import write_jsonl


def create_jsonl(args):

    # Arguments
    root = args.dataset_root
    split = args.split
    latent_dir = args.latent_dir
    out_path = args.out_path
    task = "text to audio"

    split_mapping = {
        "train": "development", 
        "test": "evaluation"
    }

    csv_path = Path(root, f"clotho_captions_{split_mapping[split]}.csv")
    meta_dict = load_meta(csv_path)

    metas = []
    paths = list(Path(latent_dir).glob("*.h5"))
    
    for n, path in enumerate(paths):
        if n % 100 == 0: 
            print(f"{n}/{len(paths)}")

        base_name = Path(path).stem.rsplit("_", 3)[0]
        prompt = meta_dict[f"{base_name}.wav"]
        attrs = get_attrs_from_hdf5(path)

        meta = {
            "task": task,
            "input": {
                "text": {
                    "prompt": prompt,
                }
            },
            "target": {
                "audio": {
                    "latent_path": str(path),
                    "latent_type": attrs["latent_type"], 
                    "fps": attrs["fps"],
                    "duration": attrs["duration"]
                }
            }
        }
        metas.append(meta)
       
    write_jsonl(metas, out_path)
    print(f"Write out to {out_path}")


def load_meta(meta_csv: str) -> dict:
    meta_dict = {}
    df = pd.read_csv(meta_csv, sep=',')

    for n in range(len(df)):
        name = df["file_name"][n]
        meta_dict[name] = [df[f"caption_{i}"][n] for i in range(1, 6)]

    return meta_dict


def get_attrs_from_hdf5(path: str) -> dict:
    with h5py.File(path, 'r') as hf:
        attrs = {
            "fps": hf.attrs["fps"],
            "duration": hf.attrs["duration"],
            "latent_type": hf.attrs["latent_type"],
        }
        
    return attrs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--latent_dir", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    create_jsonl(args)