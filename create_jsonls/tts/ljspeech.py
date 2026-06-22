import argparse
from pathlib import Path

import h5py
import pandas as pd

from audio_flow.utils import write_jsonl


def create_jsonl(args):

    # Arguments
    root = args.dataset_root
    latent_dir = args.latent_dir
    out_path = args.out_path
    task = "text to speech"

    meta_dict = load_meta(root)

    metas = []
    paths = list(Path(latent_dir).glob("*.h5"))

    for n, path in enumerate(paths):
        if n % 100 == 0: 
            print(f"{n}/{len(paths)}")

        base_name = Path(path).stem.rsplit("_", 3)[0]
        text = meta_dict[base_name]
        attrs = get_attrs_from_hdf5(path)

        meta = {
            "task": task,
            "input": {
                "text": {
                    "prompt": text,
                    "language": "en"
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


def load_meta(root: str) -> dict:
    r"""Load metadata of the GTZAN dataset."""

    # Load csv file
    csv_path = Path(root, "metadata.csv")
    df = pd.read_csv(csv_path, sep="|", header=None)
    meta_dict = {}

    for n in range(len(df)):
        name = df[0][n]
        text = df[1][n]
        meta_dict[name] = text

    return meta_dict


def get_attrs_from_hdf5(path: str) -> dict:
    with h5py.File(path, 'r') as hf:
        attrs = {
            "fps": hf.attrs["fps"],
            "duration": hf.attrs["duration"],
            "latent_type": hf.attrs["latent_type"],
            "n_frames": hf["latent"].shape[-1]
        }
        
    return attrs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    create_jsonl(args)