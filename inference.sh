#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    -i ~/.keras/datasets/flower_photos/daisy/8709110478_60d12efcd4_n.jpg \
    -c ./training_checkpoints/ckpt-8 \