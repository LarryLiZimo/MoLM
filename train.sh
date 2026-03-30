#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nproc_per_node=auto \
    train.py "$@"
