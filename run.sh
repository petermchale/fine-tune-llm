# https://huggingface.co/settings/tokens
huggingface-cli login --token $HUGGINGFACE_TOKEN
CUDA_VISIBLE_DEVICES=0 python train.py
