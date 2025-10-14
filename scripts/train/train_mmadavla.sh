# accelerate launch --config_file "./scripts/ds/1_gpu.yaml" mmadavla/train/train_mmadavla.py
# accelerate launch --config_file "./scripts/ds/1_node_1_gpus_deepspeed_zero2.yaml" mmadavla/train/train_mmadavla.py
# accelerate launch --config_file "./scripts/ds/1_node_2_gpus_deepspeed_zero2.yaml" mmadavla/train/train_mmadavla.py
accelerate launch --config_file "./scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml" mmadavla/train/train_mmadavla.py