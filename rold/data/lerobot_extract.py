import os
import torch
import shutil
import autoroot
import numpy as np
from PIL import Image
from typing import Dict
from tqdm.auto import tqdm
from functools import partial
from rold.models.magvitv2 import MagViTv2
from rold.data.utils import image_transform
from rold.data.lerobot import LeRobotDataset
from rold.models.actrvq import ActionRVQModel
from tqdm.contrib.concurrent import thread_map
from argparse import ArgumentParser, Namespace


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pretrained_visvq", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2"))
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "OpenX-LeRobot"))
    parser.add_argument("--store_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "OpenX"))
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=120)
    parser.add_argument("--num_chunks", type=int, default=24)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def process(item: Dict, vision_vq_model: MagViTv2, action_vq_model: ActionRVQModel) -> None:
    cur_image = ((item["cur_image"].transpose(0, 2)).detach().cpu().numpy() * 255).astype(np.uint8)
    cur_image = image_transform(Image.fromarray(cur_image))
    cur_image = cur_image.unsqueeze(0).to(vision_vq_model.device)
    cur_image_token = vision_vq_model.get_code(cur_image)
    cur_image_token = cur_image_token.squeeze().detach().cpu().numpy().astype(np.int16)
    pred_image = ((item["pred_image"].transpose(0, 2)).detach().cpu().numpy() * 255).astype(np.uint8)
    pred_image = image_transform(Image.fromarray(pred_image))
    pred_image = pred_image.unsqueeze(0).to(vision_vq_model.device)
    pred_image_token = vision_vq_model.get_code(pred_image)
    pred_image_token = pred_image_token.squeeze().detach().cpu().numpy().astype(np.int16)
    action = item["action"].to(dtype=torch.float, device=action_vq_model.device)
    action = action_vq_model.tokenize(action.unsqueeze(0))
    action = action.squeeze(0).detach().cpu().numpy().astype(np.int16)
    desc = item["task"] + "\n" + " ".join(map(str, item["robot_obs"].detach().cpu().numpy().tolist()))
    return {
        "desc": desc,
        "cur_image": cur_image_token,
        "pred_image": pred_image_token,
        "action": action,
    }


def main(args: Namespace) -> None:
    if args.merge:
        for dataset_name in tqdm(sorted(os.listdir(args.data_dir)), desc="Merging datasets"):
            if not os.path.exists(os.path.join(args.store_dir, dataset_name)):
                continue
            data = []
            num_chunks = set(list(map(lambda x: int(x.split("_")[-1].split(".")[0]), os.listdir(os.path.join(args.store_dir, dataset_name)))))
            if len(num_chunks) != 1:
                raise ValueError(f"{dataset_name=} {num_chunks=}")
            num_chunks = list(num_chunks)[0]
            if not all([os.path.exists(os.path.join(args.store_dir, dataset_name, f"chunk_{i}_{num_chunks}.npy")) for i in range(num_chunks)]):
                continue
            for chunk_filename in sorted(os.listdir(os.path.join(args.store_dir, dataset_name))):
                data += np.load(os.path.join(args.store_dir, dataset_name, chunk_filename), allow_pickle=True).tolist()
            np.save(os.path.join(args.store_dir, f"{dataset_name}.npy"), data)
            shutil.rmtree(os.path.join(args.store_dir, dataset_name))
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_vq_model = MagViTv2.from_pretrained(args.pretrained_visvq).to(device)
    vision_vq_model.eval()
    vision_vq_model.requires_grad_(False)
    pretrained_actrvq = os.path.join(os.getcwd(), "ckpt", f"ActRVQ_{args.action_chunk_size}steps", "92a4fa6c531aacb10c8eaa7b220e1a1f")
    action_vq_model = ActionRVQModel.from_pretrained(pretrained_actrvq).to(device)
    action_vq_model.eval()
    action_vq_model.requires_grad_(False)
    for dataset_name in sorted(os.listdir(args.data_dir)):
        if os.path.exists(os.path.join(args.store_dir, f"{dataset_name}.npy")):
            continue
        try:
            dataset = LeRobotDataset(data_dir=os.path.join(args.data_dir, dataset_name), num_chunks=args.num_chunks, chunk_idx=args.chunk_idx)
            data = thread_map(
                partial(process, vision_vq_model=vision_vq_model, action_vq_model=action_vq_model),
                dataset,
                max_workers=args.max_workers,
                desc=f"[{args.chunk_idx}/{args.num_chunks}] Processing {dataset_name}"
            )
            store_dir = os.path.join(args.store_dir, dataset_name)
            os.makedirs(store_dir, exist_ok=True)
            np.save(os.path.join(store_dir, f"chunk_{args.chunk_idx}_{args.num_chunks}.npy"), data)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue


if __name__ == "__main__":
    args = get_args()
    main(args)
