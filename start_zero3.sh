deepspeed ds_gpt2-neox.py \
    --deepspeed_config=zero3_config.json \
    --backend nccl \
    --train-dir ./data/train \
    --test-dir ./data/test \
    --val-dir ./data/val \
    --batch-size 4 \
    --epochs 8 \
    --lr 1e-4 \