# 🤖 RoLD: Robotic Large Diffusion Models

[RoLD: Robotic Large Diffusion Models](#)

[Yang Liu](https://yliu-cs.github.io), [Pengxiang Ding](https://dingpx.github.io), Tengyue Jiang, [Wenxuan Song](https://scholar.google.com/citations?user=jtFoCpwAAAAJ), Wei Zhao, [Han Zhao](https://h-zhao1997.github.io), [Donglin Wang](https://milab.westlake.edu.cn)

## 🏠 Installation

```sh
git clone https://github.com/yliu-cs/RoLD.git
conda create -n RoLD python=3.11
conda activate RoLD
cd RoLD

pip install -r requirements.txt
```

> Download [CALVIN](https://github.com/mees/calvin/tree/main/dataset), [LIBERO](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets), [Open-X Embodiment](https://huggingface.co/collections/IPEC-COMMUNITY/openx-lerobot-67c29b2ee5911f17dbea635e) and [SSV2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset).

```sh
# Extract All Actions
python rold/data/preprocess.py --action_flag --num_chunks 1

# Training Action Tokenizer
bash scripts/train/train_actrvq.sh

# Preprocess Datasets
bash scripts/data/preprocess.sh 0
bash scripts/data/preprocess.sh 8
bash scripts/data/preprocess.sh 16
```

## 🚀 Training

```sh
# Pre-Train RoLD
accelerate launch --config_file './scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml' rold/train/train_rold.py --num_train_epochs 1 --data_paths '/liuyang/Dataset/RoLD/pretrain/bridgev2_lerobot.parquet' '/liuyang/Dataset/RoLD/pretrain/calvin_abcd_8steps_pretrain.parquet'

# Fine-Tune RoLD
accelerate launch --config_file './scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml' rold/train/train_rold.py --pretrained_rold '/liuyang/LiuYang/RoLD/ckpt/RoLD/7029b3cb136e5faab45119adf8c0a927' --num_train_epochs 2 --data_paths '/liuyang/Dataset/RoLD/pretrain/calvin_abcd_8steps_pretrain.parquet'
```

## 📷 Model Checkpoint

| Model   | URL                                                      |
|---------|----------------------------------------------------------|
|   RoLD  | [HuggingFace](#)                                         |

## 🎯 Inference

```sh
python inference.py
```

## ❤️ Acknowledgment

Thanks [MMaDA](https://github.com/Gen-Verse/MMaDA), [LLaDA](https://github.com/ML-GSAI/LLaDA), [Show-o](https://github.com/showlab/Show-o), [dLLM-cache](https://github.com/maomaocun/dLLM-cache), [LLaVA-VLA](https://github.com/OpenHelix-Team/LLaVA-VLA) for their excellent code implementations, which aided later study and are referenced in this implementation as available source code.

## 📜 Citation

Please cite our paper if you use RoLD in your work:

```bibtex
@article{journals/corr/abs-xxxx-xxxxx,
  author       = {Yang Liu and Pengxiang Ding and Tengyue Jiang and Wenxuan Song and Wei Zhao and Han Zhao and Donglin Wang},
  title        = {RoLD: Robotic Large Diffusion Models},
  journal      = {CoRR},
  volume       = {abs/xxxx.xxxxx},
  year         = {2025},
}
```