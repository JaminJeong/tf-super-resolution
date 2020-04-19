#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    -i ~/.keras/datasets/flower_photos/daisy/10140303196_b88d3d6cec.jpg \
    -c ./training_checkpoints/ckpt-8 \