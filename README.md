# Audio Generation with Flow Matching

This repository contains a PyTorch implementation of audio generation with flow matching. Any modality signals, including text, audio, MIDI, image, video can be converted to audio by using conditional flow matching. The following figure shows the framework.

| Tasks                   | Supported    | Dataset    | Config yaml                                                  |
|-------------------------|--------------|------------|--------------------------------------------------------------|
| Text to music           | ✅           | GTZAN      | [configs/text2music.yaml](configs/text2music.yaml)           |
| MIDI to music           | ✅           | MAESTRO    | [configs/midi2music.yaml](configs/midi2music.yaml)           |
| Codec to audio          | ✅           | MUSDB18HQ  | [configs/codec2audio.yaml](configs/codec2audio.yaml)         |
| Mono to stereo          | ✅           | MUSDB18HQ  | [configs/mono2stereo.yaml](configs/mono2stereo.yaml)         |
| Super resolution        | ✅           | MUSDB18HQ  | [configs/superresolution.yaml](configs/superresolution.yaml) |
| Music source separation | ✅           | MUSDB18HQ  | [configs/mss.yaml](configs/mss.yaml)                         |
| Vocal to music          | ✅           | MUSDB18HQ  | [configs/vocal2music.yaml](configs/vocal2music.yaml)         |


## 0. Install dependencies

```bash
# Clone the repo
git clone https://github.com/qiuqiangkong/audio_flow
cd audio_flow

# Install Python environment
conda create --name audio_flow python=3.13

# Activate environment
conda activate audio_flow

# Install Python packages dependencies
bash env.sh
```

## 1. Download datasets

Download the dataset corresponding to the task. 

GTZAN (1.3 GB, 8 hours):

```bash
bash ./scripts/download_gtzan.sh
```

The downloaded dataset after compression looks like:

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

MUSDB18HQ (30 GB, 10 hours):

```bash
bash ./scripts/download_musdb18hq.sh
```

The downloaded dataset after compression looks like:

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

MAESTRO (131 GB, 200 hours):

```bash
bash ./scripts/download_musdb18hq.sh
```

## 2. Train

### 2.1 Train with single GPU

Here is an example of training a text to music generation system. Users can train different tasks viewing more config yaml files at [configs](configs).

```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/text2music.yaml"
```

### 2.2 Train with multiple GPUs

Users can use the Huggingface accelerate library for parallel training. train_accelerate.py just adds a few lines to train.py. Here is an example to run with 4 GPUs:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py --config="./configs/text2music.yaml"
```

## Results

Here are the results of training audio generation systems with a single RTX 4090 GPU card for 12 hours for 200k steps.

## Results

| Tasks                   | Condition    | Generated audio | Ground truth |
|-------------------------|--------------|-----------------|--------------|
| Text to music           | "blues"      | 

https://github.com/user-attachments/assets/1a62cb11-fdec-445a-a38c-3f5d9f82748e

           | N.A.         |


## Cite

```bibtex
@misc{audioflow2025,
  author       = {Qiuqiang Kong},
  title        = {AudioFlow},
  year         = {2025},
  howpublished = {\url{https://github.com/qiuqiangkong/audio_flow}},
}
```
