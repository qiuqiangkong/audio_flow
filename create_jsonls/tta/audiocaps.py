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

    csv_path = Path(root, f"{split}.csv")
    meta_dict = load_meta(csv_path)

    metas = []
    paths = list(Path(latent_dir).glob("*.h5"))

    print(meta_dict)
    # asdf
    
    for n, path in enumerate(paths):
        if n % 100 == 0: 
            print(f"{n}/{len(paths)}")

        # stem = Path(path).stem.rsplit("_", 3)[0]
        stem = Path(path).stem

        prompt = meta_dict[stem]
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
    df = pd.read_csv(meta_csv, sep=',')
    meta_dict = {}

    for n in range(len(df)):
        try:
            stem = "{}_{}".format(df["youtube_id"][n], round(df["start_time"][n]))
            meta_dict[stem] = df["caption"][n]
        except:
            pass

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