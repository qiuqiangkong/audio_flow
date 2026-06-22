import argparse

from ..mss.musdb18hq import create_jsonl


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_latent_dir", type=str, required=True)
    parser.add_argument("--target_latent_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    args.task = "codec to music"
    
    create_jsonl(args)