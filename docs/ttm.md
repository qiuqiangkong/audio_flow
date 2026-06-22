## 1. Training a Text to Music System

### 1.1 Download datasets

Download the GTZAN dataset (1.3 GB, 8 hours).

```bash
bash ./scripts/download_datasets/download_ljspeech.sh
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
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.gtzan \
        --dataset_root="./datasets/gtzan" \
        --split=${SPLIT} \
        --out_dir="./latents/gtzan/${SPLIT}/audio"
done
```

### 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m create_jsonls.ttm.gtzan \
        --latent_dir="./latents/gtzan/${SPLIT}/audio" \
        --out_path="./jsonls/ttm/${SPLIT}/gtzan.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/ttm/ttm_gtzan.yaml"
```

### 1.5 Sample
```python
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config="./configs/ttm/ttm_gtzan.yaml" \
    --ckpt_path="checkpoints/train/ttm_gtzan/step=200000_ema.pth" \
    --task="text to music" \
    --prompt="blues" \
    --out_path="out_ttm.wav"
```