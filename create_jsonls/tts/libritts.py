import argparse
from pathlib import Path

import h5py
import pandas as pd

from audio_flow.utils import write_jsonl


def create_jsonl(args):

    # Arguments
    latent_dir = args.latent_dir
    csv_path = args.csv_path
    out_path = args.out_path
    multi_jsonls = args.multi_jsonls
    chunk_size = args.chunk_size

    task = "text_to_speech"

    meta_dict = load_meta(csv_path)
    metas = []
    paths = list(Path(latent_dir).glob("*.h5"))

    for n, path in enumerate(paths):
        if n % 100 == 0: 
            print(f"{n}/{len(paths)}")

        speaker_id, chapter_id, utterance_id, segment_id = str(path.stem).split("_")[0 : 4]
        # base_name = Path(path).stem.rsplit("_", 3)[0]
        base_name = Path(path).stem
        text = meta_dict[base_name + ".wav"]
        attrs = get_attrs_from_hdf5(path)

        meta = {
            "task": task,
            "input": {
                "text": {
                    "prompt": text,
                    "language": "en",
                    "speaker_id": speaker_id
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
    
    # Sort metas by duration
    metas.sort(key=lambda meta: meta["target"]["audio"]["duration"])

    write_jsonl(metas, out_path)
    print(f"Write out to {out_path}")

    # Write to jsonls
    if multi_jsonls:
        for i in range(0, len(metas), chunk_size):
            path = Path(out_path)
            end = min(i + chunk_size, len(metas))
            path = Path(path.parent, path.stem, f"{i}-{end}{path.suffix}")

            Path(path.parent).mkdir(parents=True, exist_ok=True)
            write_jsonl(metas[i : end], path)
            print(f"Write out to {path}")
    

def load_meta(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, sep="\t")
    meta_dict = {}

    for n in range(len(df)):
        name = df["name"][n]
        content = df["text"][n]
        meta_dict[name] = content

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
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--multi_jsonls", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=10000)
    args = parser.parse_args()

    create_jsonl(args)