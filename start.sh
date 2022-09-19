export CUDA_VISIBLE_DEVICES=2,3,6,7 

python train.py --num-gpus 4 \
--train-dir ./data/train \
--test-dir ./data/test \
--val-dir ./data/val \
--seed 2021 \
--batch-size 64 \
--hidden-size 512 \
--max-seq-len 192 \
--epochs 8 \
--lr 1e-4 \
