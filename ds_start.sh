#!/bin/bash
export DEVICES=0,1,2,3
export MODEL_SIZE=33884405760
export MODEL_PARA_SIZE=8471101440
export NUM_HEAD=32
export MODEL_NAME=ds_M_4_NOCROSS_32_4096_40_1
export NUM_LAYER=40
export HIDDEN_DIM=4096
export BATCH_SIZE=64
export SEQ_LEN=512
export EXP_TYPE=ds
export GPU_SIZE_RATIO=0.45
export PARTITION_RATIO=0.45
export PARTITION_NUM=2

export CUDA_VISIBLE_DEVICES=0,1,2,3
deepspeed ds_pipeline.py \
    --deepspeed_config=pipeline_config.json \
    --backend nccl \
    --pipeline-parallel-size 4 \
    --train-dir ./data/train \
    --test-dir ./data/test \
    --val-dir ./data/val \
    --batch-size $BATCH_SIZE \
    --hidden-size 512 \
    --max-seq-len 192 \
    --epochs 8 \
    --lr 1e-4 \
    --epochs=10
