import argparse
from pathlib import Path

import h5py

from audioflow.utils.json import write_jsonl
from audioflow.utils.text import read_lines


def create_jsonl(args):

    # Arguments
    txt_dir = Path(args.input_texts_dir)
    tgt_dir = Path(args.target_latents_dir)
    out_path = Path(args.out_path)

    tgt_paths = sorted(tgt_dir.glob("*.h5"))
    metas = []

    for n, tgt_path in enumerate(tgt_paths):

        if n % 100 == 0: 
            print(f"{n}/{len(tgt_paths)}")

        # Text
        txt_path = txt_dir / f"{tgt_path.stem}.txt"

        if not txt_path.is_file():
            continue

        captions = read_lines(txt_path)
        caption = captions[0] if len(captions) > 0 else ""

        # Target latent meta
        tgt_meta = read_hdf5_attrs(tgt_path)

        meta = {
            "input": {
                "text": f"<audio>{caption}</audio>",
            },
            "target": {
                "audio": {
                    "path": tgt_meta["path"],
                    "type": tgt_meta["type"],
                    "fps": tgt_meta["fps"],
                    "duration": tgt_meta["duration"]
                }
            }
        }

        metas.append(meta)

    print(f"Total: {len(metas)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(metas, out_path)
    print(f"Write out to {out_path}")


def read_hdf5_attrs(path) -> dict:
    with h5py.File(path, "r") as hf:
        return {
            "path": str(path),
            "type": hf.attrs["type"],
            "fps": hf.attrs["fps"],
            "duration": hf.attrs["duration"]
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_texts_dir", type=str)
    parser.add_argument("--target_latents_dir", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    create_jsonl(args)