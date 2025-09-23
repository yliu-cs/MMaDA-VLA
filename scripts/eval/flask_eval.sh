#!/bin/bash

get_available_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        if [ $gpu_count -gt 0 ]; then
            gpu_ids=$(seq -s ',' 0 $((gpu_count-1)))
            echo $gpu_ids
        else
            echo "0"
        fi
    else
        echo "0"
    fi
}

gpu_list=$(get_available_gpus)
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

python mmadavla/eval/server/flask_helper.py &

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+36657))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python mmadavla/eval/server/flask_server.py --cache --port $port &
done

wait
