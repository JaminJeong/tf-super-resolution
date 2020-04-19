#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python data_augmentation.py \
    -i ~/.keras/datasets/flower_photos/daisy/9158041313_7a6a102f7a_n.jpg \
    -o ./output.jpg \