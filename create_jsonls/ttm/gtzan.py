import argparse
from pathlib import Path

import h5py

from audio_flow.utils import write_jsonl


def create_jsonl(args):

    # Arguments
    latent_dir = args.latent_dir
    out_path = args.out_path
    task = "text to music"

    paths = list(Path(latent_dir).glob("*.h5"))
    metas = []

    for n, path in enumerate(paths):
        if n % 100 == 0: 
            print(f"{n}/{len(paths)}")

        label = Path(path).stem.split(".")[0]
        attrs = get_attrs_from_hdf5(path)

        meta = {
            "task": task,
            "input": {
                "text": {
                    "prompt": label,
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
    parser.add_argument("--latent_dir", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    create_jsonl(args)