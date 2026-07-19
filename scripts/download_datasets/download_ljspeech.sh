#!/bin/bash

wget -O LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
mkdir -p ./datasets
tar -xvf LJSpeech-1.1.tar.bz2 -C ./datasets/

wget -O ./datasets/LJSpeech-1.1/train.txt https://huggingface.co/datasets/flexthink/ljspeech/resolve/main/train.txt?download=true
wget -O ./datasets/LJSpeech-1.1/valid.txt https://huggingface.co/datasets/flexthink/ljspeech/resolve/main/valid.txt?download=true
wget -O ./datasets/LJSpeech-1.1/test.txt https://huggingface.co/datasets/flexthink/ljspeech/resolve/main/test.txt?download=true