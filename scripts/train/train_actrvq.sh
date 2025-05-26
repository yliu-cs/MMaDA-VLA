NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}

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
    --num_train_epochs 50 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --logging_steps 1