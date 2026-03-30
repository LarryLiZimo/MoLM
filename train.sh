#!/bin/bash
torchrun \
    --standalone \
    --nproc_per_node=auto \
    train.py "$@"
