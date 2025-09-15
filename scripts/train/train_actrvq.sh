#!/bin/bash

if [ -f "media/flag.log" ]; then
    exit 0
else
    mkdir -p media
    touch media/flag.log
fi

LEARNING_RATES=(5e-5 1e-5)
DIMS_CONFIGS=(
    "7 1024 1024 1024 512"
    "7 2048 2048 2048 512"
)
LEVELS_CONFIGS=(
    "8 8 6 5"
    "8 8 8 6"
)

NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}

echo "开始网格搜索时间: $(date)"
echo "搜索参数配置:"
echo "Learning rates: ${LEARNING_RATES[@]}"
echo "Dims configs: ${DIMS_CONFIGS[@]}"
echo "Levels configs: ${LEVELS_CONFIGS[@]}"
echo "==========================================="

for lr in "${LEARNING_RATES[@]}"; do
    for dims in "${DIMS_CONFIGS[@]}"; do
        for levels in "${LEVELS_CONFIGS[@]}"; do
            echo "参数配置: learning_rate=$lr, dims=[$dims], levels=[$levels]"
            echo "开始时间: $(date)"
            
            torchrun \
                --nnodes=${WORLD_SIZE} \
                --node_rank=${RANK} \
                --nproc_per_node=${NGPUS} \
                --master_addr=${MASTER_ADDR} \
                --master_port=${MASTER_PORT} \
                rold/train/train_actrvq.py \
                --deepspeed scripts/ds/zero2.json \
                --output_dir ./ckpt/ActRVQ \
                --bf16 True \
                --per_device_train_batch_size 65536 \
                --eval_strategy no \
                --save_strategy epoch \
                --save_total_limit 1 \
                --save_only_model True \
                --num_train_epochs 10 \
                --learning_rate $lr \
                --lr_scheduler_type cosine \
                --warmup_ratio 0.03 \
                --logging_steps 1 \
                --dims $dims \
                --levels $levels \
                --n_steps 8
            
            echo "==========================================="
        done
    done
done

echo "Finish Grid Search: $(date)"

rm media/flag.log

# 2832