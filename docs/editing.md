## 1. Training an Audio Editing system

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

# Vocals, bass, drums, other
for STEM in "vocals" "bass" "drums" "other" do
    for SPLIT in "train" "test"; do
        CUDA_VISIBLE_DEVICES=0 python -m compute_latents.musdb18hq stereo \
            --dataset_root="./datasets/musdb18hq" \
            --stem=${STEM} \
            --split=${SPLIT} \
            --out_dir="./latents/musdb18hq/${SPLIT}/${STEM}"
    done
done
```

### 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m create_jsonls.editing.musdb18hq \
        --latent_dir="./latents/musdb18hq" \
        --split=${SPLIT} \
        --out_path="./jsonls/editing/${SPLIT}/musdb18hq.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/editing/editing_musdb18hq.yaml"
```

### 1.5 Sample
```python
for STEM in "vocals" "bass" "drums" "other"; do
    CUDA_VISIBLE_DEVICES=0 python sample.py \
        --config="./configs/editing/editing_musdb18hq.yaml" \
        --ckpt_path="checkpoints/train/editing_musdb18hq/step=200000_ema.pth" \
        --task="audio editing" \
        --prompt="separate mixture into ${STEM}" \
        --input_path="./assets/music_10s.wav" \
        --out_path="out_editing_${STEM}.wav"
done
```