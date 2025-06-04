#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
# which python
# export LD_LIBRARY_PATH=/opt/conda/envs/llava/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH

python rold/server/flask_helper.py &

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+36657))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python rold/server/flask_server.py --port $port &
done

wait
