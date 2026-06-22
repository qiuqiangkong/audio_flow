## 1. Training a Mono to Stereo System

### 1.1 Download datasets

Download the MUSDB18HQ dataset (30 GB, 9.8 hours):

```bash
bash ./scripts/download_datasets/download_musdb18hq.sh
```

The dataset structure after extraction is as follows:

<pre>
musdb18hq (30 GB)
├── train (100 files)
│   ├── A Classic Education - NightOwl
│   │   ├── bass.wav
│   │   ├── drums.wav
│   │   ├── mixture.wav
│   │   ├── other.wav
│   │   └── vocals.wav
│   ... 
│   └── ...
└── test (50 files)
    ├── Al James - Schoolboy Facination
    │   ├── bass.wav
    │   ├── drums.wav
    │   ├── mixture.wav
    │   ├── other.wav
    │   └── vocals.wav
    ... 
    └── ...
</pre>

### 1.2 Pre-extract VAE latents

```bash
# Mixture
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.musdb18hq stereo \
        --dataset_root="./datasets/musdb18hq" \
        --stem="mixture" \
        --split=${SPLIT} \
        --out_dir="./latents/musdb18hq/${SPLIT}/mixture"
done

# Mixture mono
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.musdb18hq mono \
        --dataset_root="./datasets/musdb18hq" \
        --stem="mixture" \
        --split=${SPLIT} \
        --out_dir="./latents/musdb18hq/${SPLIT}/mixture_mono"
done
```

### 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m create_jsonls.mono2stereo.musdb18hq \
        --input_latent_dir="./latents/musdb18hq/${SPLIT}/mixture_mono" \
        --target_latent_dir="./latents/musdb18hq/${SPLIT}/mixture" \
        --out_path="./jsonls/mono2stereo/${SPLIT}/musdb18hq.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/mono2stereo/mono2stereo_musdb18hq.yaml"
```

### 1.5 Sample
```python
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config="./configs/mono2stereo/mono2stereo_musdb18hq.yaml" \
    --ckpt_path="checkpoints/train/mono2stereo_musdb18hq/step=200000_ema.pth" \
    --task="mono to stereo" \
    --input_path="./assets/music_10s_mono.wav" \
    --out_path="out_mono2stereo.wav"
```