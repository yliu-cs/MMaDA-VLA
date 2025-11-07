# 🤖 MMaDA-VLA: Large Diffusion Vision-Language-Action Model with Multimodal Instruction and Generation

[MMaDA-VLA: Large Diffusion Vision-Language-Action Model with Multimodal Instruction and Generation](#)

[Yang Liu](https://yliu-cs.github.io), [Pengxiang Ding](https://dingpx.github.io), Tengyue Jiang, Wei Zhao, Minghui Lin, [Wenxuan Song](https://scholar.google.com/citations?user=jtFoCpwAAAAJ), [Han Zhao](https://h-zhao1997.github.io), [Siteng Huang](https://kyonhuang.top), [Donglin Wang](https://milab.westlake.edu.cn)

## 🏠 Installation

```sh
git clone https://github.com/yliu-cs/MMaDA-VLA.git
conda create -n MMaDA-VLA python=3.11
conda activate MMaDA-VLA
cd MMaDA-VLA

pip install -r requirements.txt
```

> Download [CALVIN](https://github.com/mees/calvin/tree/main/dataset), [LIBERO](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets), [Open-X Embodiment](https://huggingface.co/collections/IPEC-COMMUNITY/openx-lerobot-67c29b2ee5911f17dbea635e) and [SSV2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset).

```sh
# Extract All Actions
python mmadavla/data/preprocess.py --action_flag --num_chunks 1

# Normalization
python mmadavla/data/preprocess.py --norm_action

# Preprocess Datasets
bash scripts/data/preprocess.sh 0
bash scripts/data/preprocess.sh 8
bash scripts/data/preprocess.sh 16

# Merge Dataset Files
python mmadavla/data/preprocess.py --merge
```

## 🚀 Training

**Pre-Train**

```sh
accelerate launch --config_file './scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml' mmadavla/train/train_mmadavla.py --action_chunk_size 5 --save_epoch 1 --num_train_epochs 2 --data_paths '/liuyang/Dataset/MMaDA-VLA/vla_5chunk/10.parquet' '/liuyang/Dataset/MMaDA-VLA/vla_5chunk/bridgev2_lerobot.parquet' '/liuyang/Dataset/MMaDA-VLA/vla_5chunk/calvin.parquet' '/liuyang/Dataset/MMaDA-VLA/vla_5chunk/goal.parquet' '/liuyang/Dataset/MMaDA-VLA/vla_5chunk/object.parquet' '/liuyang/Dataset/MMaDA-VLA/vla_5chunk/spatial.parquet'
```

**Fine-Tune**

```sh
# CALVIN
accelerate launch --config_file './scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml' mmadavla/train/train_mmadavla.py --action_chunk_size 5 --save_epoch 1 --num_train_epochs 5 --data_paths '/liuyang/Dataset/MMaDA-VLA/vla_5chunk/calvin.parquet'

# LIBERO Spatial
accelerate launch --config_file './scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml' mmadavla/train/train_mmadavla.py --action_chunk_size 5 --save_epoch 5 --num_train_epochs 30 --data_paths '/liuyang/Dataset/MMaDA-VLA/vla_5chunk/spatial.parquet'

# LIBERO Object
accelerate launch --config_file './scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml' mmadavla/train/train_mmadavla.py --action_chunk_size 5 --save_epoch 5 --num_train_epochs 20 --data_paths '/liuyang/Dataset/MMaDA-VLA/vla_5chunk/object.parquet'

# LIBERO Goal
accelerate launch --config_file './scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml' mmadavla/train/train_mmadavla.py --action_chunk_size 5 --save_epoch 5 --num_train_epochs 30 --data_paths '/liuyang/Dataset/MMaDA-VLA/vla_5chunk/goal.parquet'

# LIBERO Long
accelerate launch --config_file './scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml' mmadavla/train/train_mmadavla.py --action_chunk_size 5 --save_epoch 5 --num_train_epochs 40 --data_paths '/liuyang/Dataset/MMaDA-VLA/vla_5chunk/10.parquet'
```

## 📷 Model Checkpoint

|     Model    | URL                                                      |
|--------------|----------------------------------------------------------|
|   MMaDA-VLA  | [HuggingFace](#)                                         |

## 🏆 Evaluation

```sh
# LIBERO
git clone https://github.com/yliu-cs/LIBERO
pip install -e LIBERO
pip install imageio[ffmpeg] robosuite==1.4.1 bddl easydict cloudpickle gym

bash scripts/eval/eval_libero.sh $libero_suite $ckpt 8 0
```

## 🎯 Inference

```sh
python inference.py
```

## ❤️ Acknowledgment

Thanks [LLaDA](https://github.com/ML-GSAI/LLaDA), [Show-o](https://github.com/showlab/Show-o), [MMaDA](https://github.com/Gen-Verse/MMaDA), [dLLM-cache](https://github.com/maomaocun/dLLM-cache), [LLaVA-VLA](https://github.com/OpenHelix-Team/LLaVA-VLA), [openpi](https://github.com/Physical-Intelligence/openpi), [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) for their excellent code implementations, which aided later study and are referenced in this implementation as available source code.

## 📜 Citation

Please cite our paper if you use MMaDA-VLA in your work:

```bibtex
@article{journals/corr/abs-xxxx-xxxxx,
  author       = {Yang Liu and Pengxiang Ding and Tengyue Jiang and Wenxuan Song and Wei Zhao and Han Zhao and Donglin Wang},
  title        = {MMaDA-VLA: Large Diffusion Vision-Language-Action Model with Multimodal Instruction and Generation},
  journal      = {CoRR},
  volume       = {abs/xxxx.xxxxx},
  year         = {2025},
}
```