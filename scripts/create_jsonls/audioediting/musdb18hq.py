import argparse
from pathlib import Path
from xml.sax.saxutils import quoteattr as qa  # Quote and escape XML attribute values

import h5py

from audioflow.utils.json import write_jsonl
from audioflow.utils.text import read_lines
from audioflow.utils.xml import check_xml


def create_jsonl(args):

    # Arguments
    root = Path(args.root)
    encoder = args.encoder
    out_path = Path(args.out_path)
    stems = ["vocals", "bass", "drums", "other"]

    # Paths
    in_dir = root / "mixture" / encoder
    in_paths = sorted(in_dir.glob("*.h5"))

    metas = []

    for n, in_path in enumerate(in_paths):
        if n % 100 == 0: 
            print(f"{n}/{len(in_paths)}")

        in_meta = read_hdf5_attrs(in_path)

        for stem in stems:
            tgt_path = root / stem / encoder / in_path.name
            tgt_meta = read_hdf5_attrs(tgt_path)

            prompt = f"separate {stem}"

            meta = {
                "input": {
                    "text": f"<audio prompt={qa(prompt)}/>",
                    "feature": {
                        "path": in_meta["path"],
                        "type": in_meta["type"],
                        "fps": in_meta["fps"],
                        "duration": in_meta["duration"]
                    }
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
            check_xml(meta["input"]["text"])
            metas.append(meta)

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
    parser.add_argument("--root", type=str)
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    create_jsonl(args)