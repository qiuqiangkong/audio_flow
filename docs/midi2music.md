## 1. Training a MIDI to Music System

### 1.1 Download datasets

Download the MAESTRO dataset (131 GB, 199 hours).

```bash
bash ./scripts/download_datasets/download_maestro.sh
```

The dataset structure after extraction is as follows:

<pre>
maestro-v3.0.0 (131 GB)
├── 2004 (132 songs, wav + flac + midi + tsv)
├── 2006 (115 songs, wav + flac + midi + tsv)
├── 2008 (147 songs, wav + flac + midi + tsv)
├── 2009 (125 songs, wav + flac + midi + tsv)
├── 2011 (163 songs, wav + flac + midi + tsv)
├── 2013 (127 songs, wav + flac + midi + tsv)
├── 2014 (105 songs, wav + flac + midi + tsv)
├── 2015 (129 songs, wav + flac + midi + tsv)
├── 2017 (140 songs, wav + flac + midi + tsv)
├── 2018 (93 songs, wav + flac + midi + tsv)
├── LICENSE
├── maestro-v3.0.0.csv
├── maestro-v3.0.0.json
└── README
</pre>

### 1.2 Pre-extract VAE latents

```bash
# Audio
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.maestro audio \
        --dataset_root="./datasets/maestro-v3.0.0" \
        --split=${SPLIT} \
        --out_dir="./latents/maestro-v3.0.0/${SPLIT}/audio"
done

# Piano roll
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.maestro midi \
        --dataset_root="./datasets/maestro-v3.0.0" \
        --split=${SPLIT} \
        --out_dir="./latents/maestro-v3.0.0/${SPLIT}/midi"
done
```

### 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m create_jsonls.midi2audio.maestro \
        --input_latent_dir="./latents/maestro-v3.0.0/${SPLIT}/midi" \
        --target_latent_dir="./latents/maestro-v3.0.0/${SPLIT}/audio" \
        --out_path="./jsonls/midi2audio/${SPLIT}/maestro-v3.0.0.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/midi2audio/midi2audio_maestro.yaml"
```

### 1.5 Sample
```python
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config="./configs/midi2audio/midi2audio_maestro.yaml" \
    --ckpt_path="checkpoints/train/midi2audio_maestro/step=200000_ema.pth" \
    --task="midi to audio" \
    --input_path="./assets/piano.mid" \
    --out_path="out_midi2audio.wav"
```