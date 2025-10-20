# 🤖 MMaDA-VLA: Robotic Manipulation Control from Multimodal Instruction and Generation

[MMaDA-VLA: Robotic Manipulation Control from Multimodal Instruction and Generation](#)

[Yang Liu](https://yliu-cs.github.io), [Pengxiang Ding](https://dingpx.github.io), Tengyue Jiang, [Wenxuan Song](https://scholar.google.com/citations?user=jtFoCpwAAAAJ), Wei Zhao, [Han Zhao](https://h-zhao1997.github.io), [Donglin Wang](https://milab.westlake.edu.cn)

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
```

## 🚀 Training

```sh
# Pre-Train MMaDA-VLA
accelerate launch --config_file './scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml' mmadavla/train/train_mmadavla.py --num_train_epochs 2 --data_paths '/liuyang/Dataset/MMaDA-VLA/vla_8chunk/bridgev2_lerobot.parquet' '/liuyang/Dataset/MMaDA-VLA/vla_8chunk/calvin.parquet'

# Fine-Tune MMaDA-VLA
accelerate launch --config_file './scripts/ds/1_node_8_gpus_deepspeed_zero2.yaml' mmadavla/train/train_mmadavla.py --pretrained_mmadavla '/liuyang/LiuYang/MMaDA-VLA/ckpt/MMaDA-VLA/...' --num_train_epochs 5 --data_paths '/liuyang/Dataset/MMaDA-VLA/vla_8chunk/calvin.parquet'
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
```

## 🎯 Inference

```sh
python inference.py
```

## ❤️ Acknowledgment

Thanks [MMaDA](https://github.com/Gen-Verse/MMaDA), [LLaDA](https://github.com/ML-GSAI/LLaDA), [Show-o](https://github.com/showlab/Show-o), [dLLM-cache](https://github.com/maomaocun/dLLM-cache), [LLaVA-VLA](https://github.com/OpenHelix-Team/LLaVA-VLA), [openpi](https://github.com/Physical-Intelligence/openpi) for their excellent code implementations, which aided later study and are referenced in this implementation as available source code.

## 📜 Citation

Please cite our paper if you use MMaDA-VLA in your work:

```bibtex
@article{journals/corr/abs-xxxx-xxxxx,
  author       = {Yang Liu and Pengxiang Ding and Tengyue Jiang and Wenxuan Song and Wei Zhao and Han Zhao and Donglin Wang},
  title        = {MMaDA-VLA: Robotic Manipulation Control from Multimodal Instruction and Generation},
  journal      = {CoRR},
  volume       = {abs/xxxx.xxxxx},
  year         = {2025},
}
```