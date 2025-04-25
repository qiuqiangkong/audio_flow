#!/bin/bash

# Download dataset
mkdir -p ./downloaded_datasets/gtzan
wget -O ./downloaded_datasets/gtzan/genres.tar.gz https://huggingface.co/datasets/qiuqiangkong/gtzan/resolve/main/genres.tar.gz?download=true

# Unzip dataset
mkdir -p ./datasets/gtzan
tar -zxvf ./downloaded_datasets/gtzan/genres.tar.gz -C ./datasets/gtzan/