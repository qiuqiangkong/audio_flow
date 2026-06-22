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

        if meta_dict[name]["tag"] == "":
            continue

        prompt = meta_dict[name]["tags_text"]
        tag = meta_dict[name]["tag"]
        
        attrs = get_attrs_from_hdf5(path)

        meta = {
            "task": task,
            "input": {
                "text": {
                    "prompt": tag,
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
            "tag": split_text(df["tags_text"].values[i])
        }

    cnt = 0
    for name in meta_dict.keys():
        if meta_dict[name]["tag"]:
            cnt += 1
    # print(meta_dict)
    # print(cnt)
    # asdf
    return meta_dict


def split_text(text):
    text = text.lower()
    text = text.split(";")[0]
    # genres = text.replace("genres:", "").split(",")
    text = text.split(": ")[1]
    tags = text.split(", ")

    tmp = []
    for tag in tags:
        if tag in KEEP_TAGS:
            tmp.append(tag)
    return " ".join(tmp)
    


KEEP_TAGS = ["rock", "pop", "jazz", "blues", "classical", "hip-hop",
    "electronic", "ambient", "folk", "country", "reggae",
    "soul", "funk", "disco", "metal", "punk"]


'''
KEEP = {
    # ===== 核心 Genre =====
    "rock", "pop", "jazz", "blues", "classical", "hip-hop",
    "electronic", "ambient", "folk", "country", "reggae",
    "soul", "funk", "disco", "metal", "punk",

    # ===== 子风格（稳定）=====
    "alternative rock", "indie rock", "progressive rock", "post-rock",
    "hard rock", "psychedelic rock", "garage rock", "rock and roll",

    "synthpop", "synthwave", "new wave", "electropop",
    "industrial", "electro", "idm", "techno", "house",
    "deep house", "tech house", "trance", "psytrance",
    "drum and bass", "jungle", "breakbeat", "dubstep",
    "downtempo", "trip-hop", "lo-fi",

    "jazz fusion", "bebop", "smooth jazz", "soul jazz",
    "nu jazz", "free jazz",

    "hip-hop instrumental", "alternative hip-hop", "trap",
    "boom bap",

    "black metal", "death metal", "thrash metal", "heavy metal",
    "metalcore", "doom metal",

    "punk rock", "hardcore punk", "post-punk",

    "bluegrass", "americana", "folk rock", "indie folk",

    "dub", "ska", "dancehall", "reggaeton",

    "afrobeat", "latin", "salsa", "tango", "bossa nova",
    "flamenco", "cumbia",

    "gospel", "choral", "opera",

    "soundtrack", "film score",

    # ===== Instrument（非常重要）=====
    "piano", "acoustic piano", "electric piano",
    "guitar", "acoustic guitar", "electric guitar",
    "bass", "double bass",
    "drums", "percussion",
    "violin", "viola", "cello", "strings",
    "flute", "clarinet", "saxophone", "trumpet", "trombone",
    "organ", "synthesizer",

    # ===== Production / texture（少量保留）=====
    "instrumental", "electronic instrumental",
    "ambient electronic", "drone", "minimal",
    "experimental", "noise", "glitch"
}
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=str)
    parser.add_argument("--parquet_path", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    create_jsonl(args)