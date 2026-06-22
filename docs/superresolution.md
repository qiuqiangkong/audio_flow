## 1. Training a Super-resolution System

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

# Mixture, low resolution
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.musdb18hq lowres \
        --dataset_root="./datasets/musdb18hq" \
        --stem="mixture" \
        --split=${SPLIT} \
        --out_dir="./latents/musdb18hq/${SPLIT}/mixture_lowres"
done
```

### 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m create_jsonls.superresolution.musdb18hq \
        --input_latent_dir="./latents/musdb18hq/${SPLIT}/mixture_lowres" \
        --target_latent_dir="./latents/musdb18hq/${SPLIT}/mixture" \
        --out_path="./jsonls/superresolution/${SPLIT}/musdb18hq.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/superresolution/superresolution_musdb18hq.yaml"
```

### 1.5 Sample
```python
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config="./configs/superresolution/superresolution_musdb18hq.yaml" \
    --ckpt_path="checkpoints/train/superresolution_musdb18hq/step=200000_ema.pth" \
    --task="super-resolution" \
    --input_path="./assets/music_10s_lowres.wav" \
    --out_path="out_superresolution.wav"
```