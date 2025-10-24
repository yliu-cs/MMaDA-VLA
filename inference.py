import os
import h5py
import torch
import autoroot
import numpy as np
from PIL import Image
from rich import print
from random import choice
from tqdm.auto import tqdm
import torch.nn.functional as F
from argparse import ArgumentParser, Namespace
from mmadavla.eval.utils import MMaDA_VLA_Server
from mmadavla.data.preprocess import merge_multiview_rgb


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--libero_data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "LIBERO-datasets"))
    parser.add_argument("--mmadavla_path", type=str, default=os.path.join(os.getcwd(), "ckpt", "MMaDA-VLA", "eedfc859ac4cfc2b2ea49a5f8a459723", "checkpoint_10"))
    parser.add_argument("--suite", type=str, default="object")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=24)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--prompt_interval_steps", type=int, default=6)
    parser.add_argument("--gen_interval_steps", type=int, default=6)
    parser.add_argument("--transfer_ratio", type=float, default=0.0)
    return parser.parse_args()


def load_libero_data(data_dir: str, action_chunk_size: int, suite: str):
    data = []
    for task in tqdm(list(map(lambda x: os.path.splitext(x)[0].replace("_demo", ""), os.listdir(os.path.join(data_dir, f"libero_{suite}")))), desc=f"Loading LIBERO {suite} tasks", ncols=100):
        task_data = h5py.File(os.path.join(data_dir, f"libero_{suite}", f"{task}_demo.hdf5"), "r")["data"]
        num_demo = sorted(list(map(lambda x: int(x.replace("demo_", "")), list(task_data.keys()))))
        for i in num_demo:
            robot_states = np.concatenate([task_data[f"demo_{i}"]["obs"]["ee_states"], task_data[f"demo_{i}"]["obs"]["gripper_states"]], axis=1)
            actions = task_data[f"demo_{i}"]["actions"]
            third_rgbs, gripper_rgbs = [task_data[f"demo_{i}"]["obs"][key] for key in ("agentview_rgb", "eye_in_hand_rgb")]
            for j in range(0, actions.shape[0]):
                k = max(j, min(j + action_chunk_size, actions.shape[0]) - 1)
                action = actions[j:k + 1]
                if action.shape[0] < action_chunk_size:
                    continue
                cur_rgb = merge_multiview_rgb(third_rgb=third_rgbs[j], gripper_rgb=gripper_rgbs[j], rgb_size=256)
                goal_rgb = merge_multiview_rgb(third_rgb=third_rgbs[min(k + 1, actions.shape[0] - 1)], gripper_rgb=gripper_rgbs[min(k + 1, actions.shape[0] - 1)], rgb_size=256)
                item = {
                    "task_inst": task.replace("_", " "),
                    "robot_states": " ".join(map(str, robot_states[j].flatten())),
                    "action": action,
                    "cur_rgb": cur_rgb,
                    "goal_rgb": goal_rgb,
                }
                data.append(item)
    return data


def main(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mmada_vla = MMaDA_VLA_Server(
        mmadavla_path=args.mmadavla_path,
        benchmark="libero",
        device=device,
        temperature=args.temperature,
        timesteps=args.timesteps, 
        cache=args.cache,
        prompt_interval_steps=args.prompt_interval_steps,
        gen_interval_steps=args.gen_interval_steps,
        transfer_ratio=args.transfer_ratio,
    )
    if not os.path.exists(os.path.join(os.getcwd(), "demo", f"{args.suite}.npy")):
        data = load_libero_data(data_dir=args.libero_data_dir, action_chunk_size=mmada_vla.training_args['action_chunk_size'], suite=args.suite)
        np.save(os.path.join(os.getcwd(), "demo", f"{args.suite}.npy"), data)
    else:
        data = np.load(os.path.join(os.getcwd(), "demo", f"{args.suite}.npy"), allow_pickle=True)
    data = choice(data)
    # data = data[327]
    task_inst, gt_action = f"{data['task_inst']}\n{data['robot_states']}", data["action"]
    # print(f"GT Tokens: {mmada_vla.action_tokenizer(gt_actions)[0]}")
    cur_rgb, goal_rgb = data["cur_rgb"], data["goal_rgb"]
    print(f"{task_inst=}")
    cur_rgb.save(os.path.join(os.getcwd(), "demo", "cur_image.jpg"))
    goal_rgb.save(os.path.join(os.getcwd(), "demo", "goal_image.jpg"))
    images, actions = mmada_vla.inference(
        task_inst=task_inst,
        merged_image=cur_rgb,
    )
    for i, (image, action) in enumerate(zip(images, actions)):
        print(f"[{i + 1}/{len(images)}] MSE: {round(F.mse_loss(torch.from_numpy(action).squeeze().to(device), torch.from_numpy(gt_action).to(device)).item(), 5)}")
        image.save(os.path.join(os.getcwd(), "demo", f"pred_image_{i}.jpg"))


if __name__ == "__main__":
    args = get_args()
    main(args)