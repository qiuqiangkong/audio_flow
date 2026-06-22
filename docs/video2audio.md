## 1. Training a Video to Audio System

### 1.1 Download datasets

Download the Audio-visual Event dataset from the official website.

The dataset structure after extraction is as follows:

<pre>
musdb18hq (5.4 GB)
├── AVE (4097 files)
│   ├── 004KfU7bgyg.mp4
│   └── ...
├── Annotations.txt
├── ReadMe.txt
├── trainSet.txt
├── testSet.txt
└── valSet.txt
</pre>

### 1.2 Pre-extract VAE latents

```bash
# Audio
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.ave audio \
        --dataset_root="./datasets/AVE" \
        --split=${SPLIT} \
        --out_dir="./latents/ave/${SPLIT}/audio"
done

# Video
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.ave video \
        --dataset_root="./datasets/AVE" \
        --split=${SPLIT} \
        --out_dir="./latents/ave/${SPLIT}/video"
done
```

### 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m create_jsonls.video2audio.ave \
        --input_latent_dir="./latents/ave/${SPLIT}/video" \
        --target_latent_dir="./latents/ave/${SPLIT}/audio" \
        --out_path="./jsonls/video2audio/${SPLIT}/ave.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/video2audio/video2audio_ave.yaml"
```

### 1.5 Sample
# TODO