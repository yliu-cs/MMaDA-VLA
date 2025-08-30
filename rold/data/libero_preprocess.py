import os
import h5py
import torch
import autoroot
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from functools import partial
from typing import Dict, List
from rold.models.magvitv2 import MagViTv2
from rold.data.utils import image_transform
from rold.models.actrvq import ActionRVQModel
from tqdm.contrib.concurrent import thread_map
from argparse import ArgumentParser, Namespace


SUITE = ["spatial", "goal", "object", "90", "10"]


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "LIBERO-datasets"))
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "LIBERO"))
    parser.add_argument("--pretrained_visvq", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2"))
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--action", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--max_workers", type=int, default=80)
    args = parser.parse_args()
    return args


def load_data(data_dir: str, action_chunk_size: int, action_flag: bool, extract_flag: bool) -> List[Dict]:
    data = [] if action_flag else ({} if extract_flag else None)
    for suite in SUITE:
        if extract_flag:
            data[suite] = []
        for task in tqdm(list(map(lambda x: os.path.splitext(x)[0].replace("_demo", ""), os.listdir(os.path.join(data_dir, f"libero_{suite}")))), desc=f"Loading {suite} tasks"):
            task_data = h5py.File(os.path.join(data_dir, f"libero_{suite}", f"{task}_demo.hdf5"), "r")["data"]
            for i in sorted(list(map(lambda x: int(x.replace("demo_", "")), list(task_data.keys())))):
                robot_states = task_data[f"demo_{i}"]["robot_states"]
                actions = task_data[f"demo_{i}"]["actions"]
                image = task_data[f"demo_{i}"]["obs"]["agentview_rgb"]
                for j in range(0, actions.shape[0]):
                    k = max(j, min(j + action_chunk_size, actions.shape[0]) - 1)
                    action = actions[j:k + 1]
                    while action.shape[0] < action_chunk_size:
                        add_action = np.zeros_like(action[-1], dtype=action.dtype)
                        add_action[-1] = action[-1][-1]
                        action = np.concatenate([action, add_action[None, :]], axis=0)
                    item = {
                        "desc": f"{task.replace('_', ' ')}\n{' '.join(map(str, robot_states[j].flatten()))}",
                        "action": action,
                        "cur_image": image[j],
                        "pred_image": image[min(k + 1, actions.shape[0] - 1)],
                    }
                    if action_flag:
                        data.append(item)
                    elif extract_flag:
                        data[suite].append(item)
    return data


def process_action(data: List[Dict], action_chunk_size: int, save_dir: str) -> None:
    actions = [item["action"] for item in data]
    actions = np.stack(actions, axis=0)
    np.save(os.path.join(save_dir, f"libero_{action_chunk_size}steps_action.npy"), actions)


def extract(item: Dict, vision_vq_model: MagViTv2, action_vq_model: ActionRVQModel) -> Dict:
    cur_image = image_transform(Image.fromarray(item["cur_image"]))
    cur_image = cur_image.unsqueeze(0).to(vision_vq_model.device)
    cur_image_token = vision_vq_model.get_code(cur_image)
    cur_image_token = cur_image_token.squeeze().detach().cpu().numpy().astype(np.int16)
    pred_image = image_transform(Image.fromarray(item["pred_image"]))
    pred_image = pred_image.unsqueeze(0).to(vision_vq_model.device)
    pred_image_token = vision_vq_model.get_code(pred_image)
    pred_image_token = pred_image_token.squeeze().detach().cpu().numpy().astype(np.int16)
    action = torch.from_numpy(item["action"]).to(dtype=torch.float, device=action_vq_model.device)
    action = action_vq_model.tokenize(action.unsqueeze(0))
    action = action.squeeze(0).detach().cpu().numpy().astype(np.int16)
    return {
        "desc": item["desc"],
        "cur_image": cur_image_token,
        "pred_image": pred_image_token,
        "action": action,
    }


def extract_vision_action(data: Dict, vision_vq_model: MagViTv2, action_vq_model: ActionRVQModel, max_workers: int, save_dir: str) -> None:
    for suite in data.keys():
        suite_data = thread_map(
            partial(extract, vision_vq_model=vision_vq_model, action_vq_model=action_vq_model),
            data[suite],
            max_workers=max_workers,
            desc=f"Extracting {suite} Tokens",
        )
        np.save(os.path.join(save_dir, f"{suite}.npy"), suite_data)


def main(args: Namespace) -> None:
    data = load_data(args.data_dir, args.action_chunk_size, args.action, args.extract)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.action:
        process_action(data, args.action_chunk_size, args.save_dir)
    if args.extract:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_vq_model = MagViTv2.from_pretrained(args.pretrained_visvq).to(device)
        vision_vq_model.eval()
        vision_vq_model.requires_grad_(False)
        pretrained_actrvq = os.path.join(os.getcwd(), "ckpt", f"ActRVQ_{args.action_chunk_size}steps", "92a4fa6c531aacb10c8eaa7b220e1a1f")
        action_vq_model = ActionRVQModel.from_pretrained(pretrained_actrvq).to(device)
        action_vq_model.eval()
        action_vq_model.requires_grad_(False)

        extract_vision_action(data, vision_vq_model, action_vq_model, args.max_workers, args.save_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)