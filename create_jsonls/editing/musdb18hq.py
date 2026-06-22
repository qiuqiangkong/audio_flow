import argparse
from pathlib import Path

import h5py

from audio_flow.utils import write_jsonl


def create_jsonl(args):

    # Arguments
    latent_dir = args.latent_dir
    split = args.split
    out_path = args.out_path
    task = args.task

    input_paths = list(Path(latent_dir, split, "mixture").glob("*.h5"))
    metas = []

    for n in range(len(input_paths)):

        if n % 100 == 0: 
            print(f"{n}/{len(input_paths)}")

        for stem in ["vocals", "bass", "drums", "other"]:
            
            prompt = f"separate mixture into {stem}"
            
            input_path = input_paths[n]
            target_path = Path(input_paths[n].parent.parent, stem, input_paths[n].name)

            if not Path(target_path).is_file():
                continue

            input_attrs = get_attrs_from_hdf5(input_path)
            target_attrs = get_attrs_from_hdf5(target_path)

            meta = {
                "task": task,
                "input": {
                    "text": {"prompt": prompt},
                    "audio": {
                        "latent_path": str(input_path),
                        "latent_type": input_attrs["latent_type"], 
                        "fps": input_attrs["fps"],
                        "duration": input_attrs["duration"]
                    }
                },
                "target": {
                    "audio": {
                        "latent_path": str(target_path),
                        "latent_type": target_attrs["latent_type"], 
                        "fps": target_attrs["fps"],
                        "duration": target_attrs["duration"]
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
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    args.task = "audio editing"

    create_jsonl(args)