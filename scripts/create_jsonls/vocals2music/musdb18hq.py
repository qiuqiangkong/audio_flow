import argparse
from scripts.create_jsonls.mss.musdb18hq import create_jsonl


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_features_dir", type=str)
    parser.add_argument("--target_latents_dir", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    create_jsonl(args)