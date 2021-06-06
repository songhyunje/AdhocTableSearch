#!/bin/bash

pip install -r requirements.txt

# TaBERT Base (K=3)
gdown 'https://drive.google.com/uc?id=1NPxbGhwJF1uU9EC18YFsEZYE-IQR7ZLj'

mkdir -p model data
tar -xvf tabert_base_k3.tar.gz -C model
rm -rf tabert_base_k3.tar.gz

# Download data (queries.txt, all.json)
