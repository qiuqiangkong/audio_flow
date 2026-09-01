# AudioFlow: Audio Generation with Flow Matching for the CCF-AATC 2026 Challenge

This repository provides a baseline for the CCF-AATC 2026 Challenge [Track 1](https://ccf-aatc.org.cn/), which aims to do music restoration under multiple distortions:
- Distortion from poor-quality amplification equipment
- Multi-speaker amplification and reverberation

## Results
The preliminary-round results of all participating teams are shown below. All numerical results are rounded to three decimal places.

| Team ID | SI-SNR (dB) ↑ | LSD ↓ | FAD ↓ | ViSQOL ↑ | Parameters (M) ↓ | MACs (G) ↓ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| T006 | -44.852 | 1.309 | 14.950 | 3.528 | 0.637 | 11.193 |
| T007 | -46.947 | 1.261 | 6.394 | 3.534 | 102.960 | 25810.000 |
| T010 | -49.067 | 1.273 | 2.540 | 3.495 | 2.951 | 82.659 |
| T021 | -49.710 | 1.858 | 4.871 | 3.284 | 441.500 | 11148.815 |
| T024 | -37.619 | 1.416 | 12.887 | 3.278 | 5.696 | 512.466 |
| T025 | -42.050 | 1.265 | 12.033 | 3.433 | 5.283 | 1148.900 |
| T026 | -39.765 | 1.448 | 3.720 | 3.406 | 1.145 | 0.933 |
| T027 | -45.583 | 1.587 | 3.694 | 3.525 | 150.750 | 2575.317 |
| T031 | -40.462 | 1.359 | 11.421 | 3.385 | 4.680 | 711.000 |
| T033 | -45.822 | 1.933 | 17.959 | 3.359 | 0.568 | 115.437 |
| T034 | -43.445 | 1.373 | 2.651 | 3.428 | 4.768 | 1592.067 |
| T042 | -51.853 | 1.766 | 3.517 | 3.361 | 476.128 | 4932.214 |
| T047 | -51.037 | 1.851 | 4.851 | 3.291 | 623.019 | 15808.892 |
| T054 | -44.529 | 1.334 | 16.345 | 3.488 | 0.662 | 1.046 |
| T059 | -46.192 | 1.773 | 19.374 | 3.415 | 4.885 | 53.590 |
| T072 | -43.252 | 1.515 | 13.770 | 3.201 | 2.340 | 187.544 |
| T074 | -45.905 | 1.388 | 2.702 | 3.333 | 2.301 | 2.109 |
| T075 | -45.764 | 1.638 | 18.807 | 3.252 | 0.472 | 4.547 |
| T088 | -40.530 | 1.269 | 2.056 | 3.446 | 2.915 | 56.437 |
| T095 | -42.812 | 1.524 | 9.191 | 3.482 | 4.994 | 160.617 |
| T098 | -38.695 | 1.363 | 17.248 | 3.424 | 0.187 | 0.486 |
| T101 | -47.561 | 1.696 | 14.390 | 2.261 | 40.832 | 1346.011 |
| T102 | -52.754 | 1.795 | 5.060 | 3.417 | 551.323 | 4637.202 |
| T106 | -47.213 | 1.102 | 0.540 | 3.786 | 364.805 | 614.993 |
| T107 | -46.148 | 3.159 | 17.231 | 2.482 | 9.660 | 1006.344 |


## 0. Install dependencies

```bash
# Install Python environment
conda create --name audio_flow python=3.10

# Activate environment
conda activate audio_flow
cd audio_flow

# Install Python packages dependencies
bash env.sh
```

## 1. Baseline Training

### 1.1 Download datasets

Download the officially provided [dataset](https://ccf-aatc.org.cn/) (10 hours):

The dataset structure is as follows:

<pre>
dataset
├── train (202 files)
│   ├── 安静街道__笔记本_低_1M
│   │   ├── 原始.wav
│   │   ├── pcm01.wav
│   │   ├── pcm02.wav
│   │   ├── pcm03.wav
│   │   └── phone.wav
│   ... 
│   └── ...
└── valid (25 files)
    ├── 户外公园__平板_高_5M
    │   ├── 原始.wav
    │   ├── pcm01.wav
    │   ├── pcm02.wav
    │   ├── pcm03.wav
    │   └── phone.wav
    ... 
    └── ...
</pre>
The `原始.wav` file is the target audio. `phone.wav` is the degraded audio recorded with a mobile phone and requires restoration. `pcm01/02/03.wav` are degraded recordings captured by three microphones. The three microphones are arranged in a linear array, with microphone spacings of 40 mm and 120 mm. The microphone recordings are sampled at 48 kHz, while all other audio files are sampled at 16 kHz. The microphone recordings can be used as additional training data to augment the dataset. In the baseline setting, `phone.wav` is used as the input audio, and the `原始.wav` file is used as the target.

### 1.2 Test set

The test set is provided for final evaluation and contains mobile-phone recordings collected under multiple acoustic scenarios. Unlike the training and validation sets, the test set only provides the degraded audio `phone.wav` for each scene. The final evaluation is conducted only on `phone.wav`.

### 1.3 Pre-extract VAE latents

```bash
# Mixture
for SPLIT in "train" "valid"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.musdb18hq stereo \
        --dataset_root="./datasets" \
        --stem="phone" \
        --split=${SPLIT} \
        --latent_type="mmaudio_vae" \
        --out_dir="./latents/aatc/${SPLIT}/mixture"
done

# Target
for SPLIT in "train" "valid"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.musdb18hq stereo \
        --dataset_root="./datasets" \
        --stem="原始" \
        --split=${SPLIT} \
        --latent_type="mmaudio_vae" \
        --out_dir="./latents/aatc/${SPLIT}/target"
done
```

### 1.4 Prepare JSONL files

```bash
for SPLIT in "train" "valid"; do
    python -m create_jsonls.mss.musdb18hq \
        --input_latent_dir="./latents/aatc/${SPLIT}/mixture" \
        --target_latent_dir="./latents/aatc/${SPLIT}/target" \
        --out_path="./jsonls/mss/${SPLIT}/aatc.jsonl"
done
```

### 1.5 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/mss/mss_musdb18hq.yaml"
```

### 1.6 Sample
The pretrained checkpoint can be downloaded from the GitHub Release page: [Download checkpoint](https://github.com/fliu215/audioflow/releases/download/ckpt-v1/step.1000000_ema.pth)
```python
# Single
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config="./configs/mss/mss_musdb18hq.yaml" \
    --ckpt_path="checkpoints/train/mss_musdb18hq/step=1000000_ema.pth" \
    --task="music source separation" \
    --duration=10 \
    --input_path="./assets/music_10s.wav" \
    --out_path="out_mss.wav"
# Batch
CUDA_VISIBLE_DEVICES=0 python batch_sample_chunked.py \
  --config "./configs/mss/mss_musdb18hq.yaml" \
  --ckpt_path "checkpoints/train/mss_musdb18hq/step=1000000_ema.pth" \
  --dataset_root "datasets/test" \
  --input_filename "phone" \
  --out_dir "batch_results" \
  --chunk_duration "10" \
  --overlap_duration "2" \
  --output_sr "16000" \
  --mono_output \
  --skip_existing
```

### 1.7 Evaluate
The command to run the quality evaluation metric calculation script is as follows:
```
python evaluate_mss_metrics.py --compute_fad --compute_visqol --compute_input_metrics
```

The command to run the baseline model complexity calculation script is as follows:
```
CUDA_VISIBLE_DEVICES=0 python evaluate_complexity.py \
  --config ./configs/mss/mss_musdb18hq.yaml \
  --ckpt-path checkpoints/train/mss_musdb18hq/step=1000000_ema.pth \
  --solver-steps 100 \
  --device cuda \
  --json-out complexity_report.json
```
or
```
python evaluate_complexity.py \
  --config ./configs/mss/mss_musdb18hq.yaml \
  --solver-steps 100 \
  --device cpu \
  --json-out complexity_report.json
```

The validation results of the baseline system are shown below.

| Method | SI-SNR (dB) ↑ | LSD ↓ | FAD ↓ | ViSQOL ↑ | Para. ↓ | MACs ↓
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | -46.91 | 1.91 | 22.70 | 3.37 | - | - |
| Baseline | -50.50 | 1.71 | 8.90 | 3.45 | 550.87M | 4.49T |

Note:The reported parameter count and MACs cover the complete end-to-end inference pipeline, including VAE encoding, condition processing, all iterative denoising steps, and VAE/BigVGAN decoding. For 100 solver steps, all 99 denoiser evaluations are included.

## External links

[1] Conditional flow matching: https://github.com/atong01/conditional-flow-matching

[2] DiT: https://github.com/facebookresearch/DiT

[3] Contributor: https://fliu215.github.io/homepage/
