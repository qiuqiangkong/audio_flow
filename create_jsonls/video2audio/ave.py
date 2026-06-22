import argparse
from pathlib import Path

import h5py
import pandas as pd

from audio_flow.utils import write_jsonl


def create_jsonl(args):

    # Arguments
    text_dir = args.text_dir
    input_latent_dir = args.video_latent_dir
    target_latent_dir = args.audio_latent_dir
    out_path = args.out_path
    task = args.task

    input_paths = list(Path(input_latent_dir).glob("*.h5"))
    target_paths = list(Path(target_latent_dir).glob("*.h5"))

    input_names = [path.name for path in input_paths]
    target_names = [path.name for path in target_paths]
    names = intersect_lists(input_names, target_names)

    input_paths = [Path(input_latent_dir, name) for name in names]
    target_paths = [Path(target_latent_dir, name) for name in names]
    metas = []

    for n in range(len(names)):
        if n % 100 == 0: 
            print(f"{n}/{len(names)}")

        input_attrs = get_attrs_from_hdf5(input_paths[n])
        target_attrs = get_attrs_from_hdf5(target_paths[n])

        text_path = Path(text_dir, Path(names[n]).with_suffix(".txt"))
        df = pd.read_csv(text_path, sep='\t', header=None)
        caption = df[0].values[0]

        meta = {
            "task": task,
            "input": {
                "text": {
                    "prompt": caption,
                },
                "video": {
                    "latent_path": str(input_paths[n]),
                    "latent_type": input_attrs["latent_type"], 
                    "fps": input_attrs["fps"],
                    "duration": input_attrs["duration"]
                }
            },
            "target": {
                "audio": {
                    "latent_path": str(target_paths[n]),
                    "latent_type": target_attrs["latent_type"], 
                    "fps": target_attrs["fps"],
                    "duration": target_attrs["duration"]
                }
            }
        }
        metas.append(meta)

    write_jsonl(metas, out_path)
    print(f"Write out to {out_path}")


def intersect_lists(list1, list2) -> list:
    return [x for x in list1 if x in set(list2)]


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
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--video_latent_dir", type=str, required=True)
    parser.add_argument("--audio_latent_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    args.task = "video to audio"
    
    create_jsonl(args)