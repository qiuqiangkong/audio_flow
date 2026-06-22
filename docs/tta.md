## 1. Training a Text to Audio System

### 1.1 Download datasets

Download the AudioCaps dataset (131 GB) from the official website.

The dataset structure after extraction is as follows:

<pre>
audiocaps (131 GB)
├── train (49274 files)
├── val (494 files)
├── test (957 files)
├── train.csv
├── val.csv
├── test.csv
├── LICENSE.txt
└── README.md
</pre>

### 1.2 Pre-extract VAE latents

```bash
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.audiocaps \
        --dataset_root="./datasets/audiocaps2.0" \
        --split=${SPLIT} \
        --out_dir="./latents/audiocaps2.0/${SPLIT}/audio"
done
```

### 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m create_jsonls.tta.audiocaps \
        --dataset_root="./datasets/audiocaps2.0" \
        --split=${SPLIT} \
        --latent_dir="./latents/audiocaps2.0/${SPLIT}/audio" \
        --out_path="./jsonls/tta/${SPLIT}/audiocaps2.0.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/tta/tta_audiocaps.yaml"
```

### 1.5 Sample
```python
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config="./configs/tta/tta_audiocaps.yaml" \
    --ckpt_path="checkpoints/train/tta_audiocaps/step=200000_ema.pth" \
    --task="text to audio" \
    --prompt="a dog barking and a children speaking." \
    --out_path="out_tta.wav"
```