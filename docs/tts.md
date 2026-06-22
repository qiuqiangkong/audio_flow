## 1. Training a Text to Speech System using LJSpeech Dataset

### 1.1 Download datasets

Download the LJSpeech dataset corresponding to the task. 

```bash
bash ./scripts/download_datasets/download_ljspeech.sh
```

The dataset structure after extraction is as follows:

<pre>
LJSpeech-1.1 (3.6 GB)
├── wavs (13,100 .wavs)
│   ├── LJ001-0001.wav
│   └── ...
├── metadata.csv
├── README
├── train.txt
├── valid.txt
└── test.txt
</pre>

### 1.2 Pre-extract VAE latents

```bash
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.ljspeech \
        --dataset_root="./datasets/LJSpeech-1.1" \
        --split=${SPLIT} \
        --out_dir="./latents/ljspeech/${SPLIT}/audio"
done
```

### 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m create_jsonls.tts.ljspeech \
        --dataset_root="./datasets/LJSpeech-1.1" \
        --latent_dir="./latents/ljspeech/${SPLIT}/audio" \
        --out_path="./jsonls/tts/${SPLIT}/ljspeech.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/tts/tts_ljspeech.yaml"
```

### 1.5 Sample
```python
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config="./configs/tts/tts_ljspeech.yaml" \
    --ckpt_path="checkpoints/train/tts_ljspeech/step=200000_ema.pth" \
    --task="text to speech" \
    --prompt="Today is a sunny day." \
    --out_path="out_tts.wav"
```

## 2. LibriTTS

### 1.1 Download datasets

Download the LibriTTS dataset corresponding to the task. 

The dataset structure after extraction is as follows:

<pre>
LibriTTS
├── train-clean-100
├── train-clean-360
├── train-other-500
├── test-clean
├── test-other
└── ...
</pre>

### 1.2 Pre-extract VAE latents

```bash
for SUBDIR in "train-clean-100" "train-clean-360" "train-other-500"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.libritts \
        --audios_dir="./datasets/LibriTTS/${SUBDIR}" \
        --out_dir="./latents/libritts/train/${SUBDIR}/audio" \
        --csv_path="./latents/libritts/train/${SUBDIR}/text.csv"
done

for SUBDIR in "test-clean" "test-other"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.libritts \
        --audios_dir="./datasets/LibriTTS/${SUBDIR}" \
        --out_dir="./latents/libritts/test/${SUBDIR}/audio" \
        --csv_path="./latents/libritts/test/${SUBDIR}/text.csv"
done
```

### 1.3 Prepare JSONL files

```bash
SPLIT="train"
for SUBDIR in "train-clean-100" "train-clean-360" "train-other-500"; do
    python -m create_jsonls.tts.libritts \
        --latent_dir="./latents/libritts/${SPLIT}/${SUBDIR}/audio" \
        --csv_path="./latents/libritts/${SPLIT}/${SUBDIR}/text.csv" \
        --out_path="./jsonls/tts/${SPLIT}/libritts_${SUBDIR}.jsonl" \
        --multi_jsonls \
        --chunk_size=10000
done

SPLIT="test"
for SUBDIR in "test-clean" "test-other"; do
    python -m create_jsonls.tts.libritts \
        --latent_dir="./latents/libritts/${SPLIT}/${SUBDIR}/audio" \
        --csv_path="./latents/libritts/${SPLIT}/${SUBDIR}/text.csv" \
        --out_path="./jsonls/tts/${SPLIT}/libritts_${SUBDIR}.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/tts/tts_libritts.yaml"
```

### 1.5 Sample
```python
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config="./configs/tts/tts_libritts.yaml" \
    --ckpt_path="checkpoints/train/tts_libritts/step=200000_ema.pth" \
    --task="text to speech" \
    --prompt="Today is a sunny day." \
    --out_path="out_tts.wav"
```