import argparse
import pandas as pd
from pathlib import Path

import h5py

from audio_flow.utils import write_jsonl


def create_jsonl(args):

    # Arguments
    latent_dir = args.latent_dir
    parquet_path = args.parquet_path
    out_path = args.out_path
    task = "text to music"

    meta_dict = load_meta(parquet_path)

    paths = list(Path(latent_dir).glob("*.h5"))
    metas = []

    for n, path in enumerate(paths):
        if n % 100 == 0: 
            print(f"{n}/{len(paths)}")

        name = Path(path).stem
        if name not in meta_dict:
            continue

        prompt = meta_dict[name]["tags_text"]
        
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
    print(f"Number: {len(metas)}")
    print(f"Write out to {out_path}")


def get_attrs_from_hdf5(path: str) -> dict:
    with h5py.File(path, 'r') as hf:
        attrs = {
            "fps": hf.attrs["fps"],
            "duration": hf.attrs["duration"],
            "latent_type": hf.attrs["latent_type"],
        }
        
    return attrs


def load_meta(parquet_path: str) -> dict:
    df = pd.read_parquet(parquet_path)

    meta_dict = {}
    for i in range(len(df)):
        name = df["id"].values[i]
        meta_dict[name] = {
            "chatgpt_text": df["chatgpt_texts"].values[i],
            "tags_text": df["tags_text"].values[i],
        }

    return meta_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=str)
    parser.add_argument("--parquet_path", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    create_jsonl(args)