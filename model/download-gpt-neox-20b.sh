#!/bin/bash
for i in $(seq -f '%05.f' 46)
do
   echo "Start Downloading $i slice."
   wget "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/pytorch_model-$i-of-00046.bin"
done