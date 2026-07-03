# AudioFlow: Audio Generation with Flow Matching

This repository contains a tutorial on audio generation using conditional flow matching implemented in PyTorch. Signals from any modality, including text, audio, MIDI, images, and video, can be converted to audio using conditional flow matching. The figure below shows the framework.

The supported tasks include:

| Tasks                   | Supported    | Dataset    | How to run                                                  |
|-------------------------|--------------|------------|--------------------------------------------------------------|
| Text to music           | ✅           | GTZAN      | [docs/ttm.md](docs/ttm.md)                             |
| Text to speech          | ✅           | LJSpeech   | [docs/tts.md](docs/tts.md)                           |
| Text to audio           | ✅           | AudioCaps  | [docs/tta.md](docs/tta.md)                            |
| Music source separation | ✅           | MUSDB18HQ  | [docs/mss.md](docs/mss.md)                             |
| Mono to stereo          | ✅           | MUSDB18HQ  | [docs/mono2stereo.md](docs/mono2stereo.md)             |
| Super resolution        | ✅           | MUSDB18HQ  | [docs/superresolution.md](docs/superresolution.md)     |
| Vocals to music         | ✅           | MUSDB18HQ  | [docs/vocals2music.md](docs/vocals2music.md)       |
| Audio editing           | ✅           | MUSDB18HQ  | [docs/editing.md](docs/editing.md)       |
| MIDI to music           | ✅           | MAESTRO    | [docs/midi2music.md](docs/midi2music.md)  |
| Video to audio          | ✅           | Audio-Visual Event    | [docs/video2audio.md](docs/video2audio.md)                                                    |

## 0. Install dependencies

```bash
# Clone the repo
git clone https://github.com/qiuqiangkong/audioflow
cd audioflow

# Install packages
uv sync

# Activate env
source .venv/bin/activate
```

## 1. Train a text to music system

See [docs](docs) for more examples.

### 1.1 Download datasets

Download the GTZAN dataset (1.3 GB, 8 hours).

```bash
bash ./scripts/download_datasets/download_gtzan.sh
```

The dataset structure after extraction is as follows:

<pre>
gtzan (1.3 GB)
└── genres
    ├── blues (100 files)
    ├── classical (100 files)
    ├── country (100 files)
    ├── disco (100 files)
    ├── hiphop (100 files)
    ├── jazz (100 files)
    ├── metal (100 files)
    ├── pop (100 files)
    ├── reggae (100 files)
    └── rock (100 files)
</pre>

### 1.2 Pre-extract VAE latents

```bash
# Extract audio latent
ENCODER="levo_vae"
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m scripts.extract_features.gtzan audio \
        --dataset_root="./datasets/gtzan" \
        --split=${SPLIT} \
        --encoder_name=${ENCODER} \
        --out_dir="./features/gtzan/${SPLIT}/audio/${ENCODER}"
done

# Extract texts
for SPLIT in "train" "test"; do
    python -m scripts.extract_features.gtzan text \
        --dataset_root="./datasets/gtzan" \
        --split=${SPLIT} \
        --out_dir="./features/gtzan/${SPLIT}/text/raw"
done
```
```

### 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m scripts.create_jsonls.ttm.gtzan \
        --input_texts_dir="./features/gtzan/${SPLIT}/text/raw" \
        --target_latents_dir="./features/gtzan/${SPLIT}/audio/levo_vae" \
        --out_path="./jsonls/ttm/${SPLIT}/gtzan.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config="./configs/ttm/ttm_gtzan.yaml"
```

### 2. Sample
```python
CUDA_VISIBLE_DEVICES=0 python -m scripts.sample \
    --config="./configs/ttm/ttm_gtzan.yaml" \
    --ckpt_path="checkpoints/train/ttm_gtzan/step=200000_ema.pth" \
    --prompt="<music>text to music</music>" \
    --out_path="gen_ttm.wav"
```

## Cite
```bibtex
@inproceedings{midi2symphony,
  author = {Jiahe Lei, Qiuqiang Kong},
  title  = {Symphony Rendering: Midi and Composer-Conditioned Auto Orchestration with Flow-Matching Transformers},
  year   = {2026},
  booktitle    = {International Conference on Acoustics, Speech, and Signal Processing (ICASSP)}
}

@article{audioflow,
  author = {To appear},
}
```

## External links

[1] Conditional flow matching: https://github.com/atong01/conditional-flow-matching

[2] DiT: https://github.com/facebookresearch/DiT