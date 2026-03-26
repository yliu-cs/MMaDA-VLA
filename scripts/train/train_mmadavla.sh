#!/bin/bash

# accelerate launch --config_file "./scripts/ds/1_gpu.yaml" mmadavla/train/train_mmadavla.py
# accelerate launch --config_file "./scripts/ds/1_node_1_gpus_deepspeed_zero2.yaml" mmadavla/train/train_mmadavla.py
# accelerate launch --config_file "./scripts/ds/1_node_2_gpus_deepspeed_zero2.yaml" mmadavla/train/train_mmadavla.py


accelerate launch --config_file "./scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml" \
    mmadavla/train/train_mmadavla.py \
    --pretrained_mmadavla './ckpt/MMaDA-VLA/PreTrained' \
    --action_chunk_size 10 \
    --save_epoch 2 \
    --num_train_epochs 2 \
    --output_dir './ckpt/MMaDA-VLA/calvin_abc_d_wo_pad_10pct' \
    --data_paths '/liuyang/Dataset/MMaDA-VLA/vla_10chunk/bak/calvin_abc_d_without_pad_10pct.parquet'