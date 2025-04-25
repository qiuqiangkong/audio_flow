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

MUSDB18HQ (30 GB, 10 hours):

```bash
bash ./scripts/download_musdb18hq.sh
```

MAESTRO (131 GB, 200 hours):

```bash
bash ./scripts/download_musdb18hq.sh
```

## 1. Text to music

### 1.1 Download dataset

Users need to do download the GTZAN dataset (1.3 GB, 8 hours).



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


### 1.2 Train

Takes \~5 hours on 1 RTX4090 to train for 100,000 steps.

```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/text2music.yaml"
```

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py --config="./configs/text2music.yaml"
```











## 1. Download dataset

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

## 2. Train

Takes \~3 hours on 1 RTX4090 to train for 100,000 steps.

```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/unet.yaml"
```

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py --config="./configs/unet.yaml"
```

## 3. Inference

## 4. Evaluate

