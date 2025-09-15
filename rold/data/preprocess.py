import os
import json
import h5py
import torch
import autoroot
import importlib
import numpy as np
import polars as pl
from PIL import Image
from enum import Enum
from tqdm.auto import tqdm
from itertools import chain
from functools import partial
from rold.utils.prompt import ignore_id
from rold.models.magvitv2 import MagViTv2
from typing import List, Dict, Tuple, Union
from rold.data.utils import image_transform
from rold.data.lerobot import LeRobotDataset
from rold.models.actrvq import ActionRVQModel
from tqdm.contrib.concurrent import thread_map
from argparse import ArgumentParser, Namespace
from rold.utils.misc import get_chunk, freeze_module


class ROBOTDATASET(Enum):
    calvin = 1
    libero = 2
    oxe = 3
    ssv2 = 4


def get_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--calvin_data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "CALVIN"))
    parser.add_argument("--libero_data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "LIBERO-datasets"))
    parser.add_argument("--oxe_data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "OpenX-LeRobot"))
    parser.add_argument("--ssv2_data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "SSV2"))
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "RoLD", "preprocessed"))
    parser.add_argument("--pretrained_visvq", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2"))
    parser.add_argument("--pretrained_actrvq", type=str, default="2a3076dddc359e0d84989b550b36e27a")
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--action_flag", action="store_true")
    parser.add_argument("--num_chunks", type=int, default=24)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=80)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


def merge_multiview_rgb(third_rgb: np.ndarray, gripper_rgb: np.ndarray = None, rgb_size: int = 256) -> Image.Image:
    third_rgb = Image.fromarray(third_rgb)
    if gripper_rgb is not None:
        gripper_rgb = Image.fromarray(gripper_rgb)
    third_rgb = third_rgb.resize((rgb_size, rgb_size // 2), Image.LANCZOS)
    if gripper_rgb is not None:
        gripper_rgb = gripper_rgb.resize((rgb_size, rgb_size // 2), Image.LANCZOS)
    rgb = Image.new("RGB", (rgb_size, rgb_size))
    rgb.paste(third_rgb, (0, 0))
    if gripper_rgb is not None:
        rgb.paste(gripper_rgb, (0, rgb_size // 2))
    return rgb


def extract_rgb_token(rgb: Image.Image, rgb_vq_model: MagViTv2) -> np.ndarray:
    rgb = image_transform(rgb).unsqueeze(0).to(rgb_vq_model.device)
    rgb_token = rgb_vq_model.get_code(rgb)
    rgb_token = rgb_token.squeeze().detach().cpu().numpy().astype(np.int16)
    return rgb_token


def extract_action_token(action: np.ndarray, action_vq_model: ActionRVQModel) -> np.ndarray:
    action = torch.from_numpy(action).to(dtype=torch.float, device=action_vq_model.device)
    action_token = action_vq_model.tokenize(action.unsqueeze(0))
    action_token = action_token.squeeze(0).detach().cpu().numpy().astype(np.int16)
    return action_token


def extract_tokens(item: Dict, rgb_vq_model: MagViTv2, action_vq_model: ActionRVQModel) -> Dict:
    cur_rgb, goal_rgb = [extract_rgb_token(rgb=item[key], rgb_vq_model=rgb_vq_model) for key in ("cur_rgb", "goal_rgb")]
    action = extract_action_token(action=item["action"], action_vq_model=action_vq_model)
    return {
        "task_inst": item["task_inst"],
        "robot_states": item["robot_states"],
        "action": action,
        "cur_rgb": cur_rgb,
        "goal_rgb": goal_rgb,
    }


def convert_polars(data: Union[List, Dict], action_chunk_size: int, num_quantizers: int) -> pl.DataFrame:
    data = pl.DataFrame(data)
    data = data.with_columns([
        pl.col("cur_rgb").map_elements(lambda x: x.flatten().tolist()).cast(pl.List(pl.Int16)),
        pl.col("goal_rgb").map_elements(lambda x: x.flatten().tolist()).cast(pl.List(pl.Int16)),
    ])
    data = data.with_columns([
        pl.col("action").map_elements(lambda x: x.tolist()).cast(pl.List(pl.Int16)),
    ]) if "action" in data.columns else data.with_columns([
        pl.lit([ignore_id] * action_chunk_size * num_quantizers).cast(pl.List(pl.Int16)).alias("action"),
    ])
    return data


def extract_calvin(
    data_dir: str,
    action_chunk_size: int,
    rgb_vq_model: MagViTv2,
    action_vq_model: ActionRVQModel,
    action_flag: bool,
    num_chunks: int,
    chunk_idx: int,
    max_workers: int,
    debug: bool
) -> None:
    task = "ABCD_D"  # ["ABC_D", "ABCD_D"]
    data_dir = os.path.join(data_dir, f"task_{task.upper()}", "training")
    language_ann_data = np.load(os.path.join(data_dir, "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True).item()
    indx, ann = language_ann_data["info"]["indx"], language_ann_data["language"]["ann"]
    def get_action_chunk(s: int, e: int) -> np.ndarray:
        action_chunk = []
        for i in range(action_chunk_size):
            if s + i < e + 1:
                action_chunk.append(np.load(os.path.join(data_dir, f"episode_{str(s + i).zfill(7)}.npz"))["rel_actions"])
        while len(action_chunk) < action_chunk_size:
            add_act = np.zeros_like(action_chunk[-1], dtype=action_chunk[-1].dtype)
            add_act[-1] = action_chunk[-1][-1]
            action_chunk.append(add_act)
        return np.stack(action_chunk)
    def calvin_process(t: Tuple[int, int], desc: str) -> List[Dict]:
        s, e = t
        if not all([os.path.exists(os.path.join(data_dir, f"episode_{str(i).zfill(7)}.npz")) for i in range(s, e + 1)]):
            return []
        data = []
        for i in range(s, e + 1):
            if not action_flag:
                cur_episode = np.load(os.path.join(data_dir, f"episode_{str(i).zfill(7)}.npz"))
                goal_episode = np.load(os.path.join(data_dir, f"episode_{str(min(i + action_chunk_size, e)).zfill(7)}.npz"))
                cur_third_rgb, cur_gripper_rgb = cur_episode["rgb_static"], cur_episode["rgb_gripper"]
                goal_third_rgb, goal_gripper_rgb = goal_episode["rgb_static"], goal_episode["rgb_gripper"]
                cur_rgb = merge_multiview_rgb(third_rgb=cur_third_rgb, gripper_rgb=cur_gripper_rgb, rgb_size=256)
                goal_rgb = merge_multiview_rgb(third_rgb=goal_third_rgb, gripper_rgb=goal_gripper_rgb, rgb_size=256)
            action = get_action_chunk(i, min(i + action_chunk_size - 1, e))
            action[..., -1] = np.clip(action[..., -1], 0, 1)
            if action_flag:
                data.append(action)
            else:
                data.append(
                    extract_tokens(
                        item={
                            "desc": desc,
                            "robot_states": " ".join(map(str, cur_episode["robot_obs"].flatten())),
                            "action": action,
                            "cur_rgb": cur_rgb,
                            "goal_rgb": goal_rgb,
                        },
                        rgb_vq_model=rgb_vq_model,
                        action_vq_model=action_vq_model
                    )
                )
        return data
    indx, ann = get_chunk(indx, n=num_chunks, k=chunk_idx), get_chunk(ann, n=num_chunks, k=chunk_idx)
    if debug:
        indx, ann = indx[:2], ann[:2]
    data = thread_map(
        calvin_process,
        indx,
        ann,
        max_workers=max_workers,
        desc=f"[{chunk_idx}/{num_chunks}] Construct CALVIN data",
        ncols=100,
    )
    data = list(chain(*data))
    if not action_flag:
        data = convert_polars(data=data, action_chunk_size=action_chunk_size, num_quantizers=action_vq_model.config.num_quantizers)
    return data


def extract_libero(
    data_dir: str,
    action_chunk_size: int,
    rgb_vq_model: MagViTv2,
    action_vq_model: ActionRVQModel,
    action_flag: bool,
    num_chunks: int,
    chunk_idx: int,
    max_workers: int,
    debug: bool
) -> List[Dict]:
    SUITE = ["spatial", "goal", "object", "90", "10"]
    data = {}
    for suite in SUITE:
        data[suite] = []
        for task in tqdm(list(map(lambda x: os.path.splitext(x)[0].replace("_demo", ""), os.listdir(os.path.join(data_dir, f"libero_{suite}")))), desc=f"Loading LIBERO {suite} tasks", ncols=100):
            task_data = h5py.File(os.path.join(data_dir, f"libero_{suite}", f"{task}_demo.hdf5"), "r")["data"]
            num_demo = sorted(list(map(lambda x: int(x.replace("demo_", "")), list(task_data.keys()))))
            if debug:
                num_demo = num_demo[:2]
            for i in num_demo:
                robot_states = np.concatenate([task_data[f"demo_{i}"]["obs"]["ee_states"], task_data[f"demo_{i}"]["obs"]["gripper_states"]], axis=1)
                actions = task_data[f"demo_{i}"]["actions"]
                third_rgbs, gripper_rgbs = [task_data[f"demo_{i}"]["obs"][key] for key in ("agentview_rgb", "eye_in_hand_rgb")]
                for j in range(0, actions.shape[0]):
                    k = max(j, min(j + action_chunk_size, actions.shape[0]) - 1)
                    action = actions[j:k + 1]
                    while action.shape[0] < action_chunk_size:
                        add_action = np.zeros_like(action[-1], dtype=action.dtype)
                        add_action[-1] = action[-1][-1]
                        action = np.concatenate([action, add_action[None, :]], axis=0)
                    action[..., -1] = 1 - np.clip(action[..., -1], 0, 1)
                    if not action_flag:
                        cur_rgb = merge_multiview_rgb(third_rgb=third_rgbs[j], gripper_rgb=gripper_rgbs[j], rgb_size=256)
                        goal_rgb = merge_multiview_rgb(third_rgb=third_rgbs[min(k + 1, actions.shape[0] - 1)], gripper_rgb=gripper_rgbs[min(k + 1, actions.shape[0] - 1)], rgb_size=256)
                        item = {
                            "task_inst": task.replace("_", " "),
                            "robot_states": " ".join(map(str, robot_states[j].flatten())),
                            "action": action,
                            "cur_rgb": cur_rgb,
                            "goal_rgb": goal_rgb,
                        }
                    data[suite].append(action if action_flag else item)
    if action_flag:
        data = list(chain(*list(data.values())))
    else:
        for suite in SUITE:
            data[suite] = thread_map(
                partial(extract_tokens, rgb_vq_model=rgb_vq_model, action_vq_model=action_vq_model),
                get_chunk(data[suite], n=num_chunks, k=chunk_idx),
                max_workers=max_workers,
                desc=f"[{chunk_idx}/{num_chunks}] Extract LIBERO Tokens",
                ncols=100,
            )
            data[suite] = convert_polars(data=data[suite], action_chunk_size=action_chunk_size, num_quantizers=action_vq_model.config.num_quantizers)
    return data


def extract_oxe(
    data_dir: str,
    action_chunk_size: int,
    rgb_vq_model: MagViTv2,
    action_vq_model: ActionRVQModel,
    action_flag: bool,
    num_chunks: int,
    chunk_idx: int,
    max_workers: int,
    debug: bool
) -> None:
    DATASETS = [
        "ucsd_kitchen_dataset_lerobot",
        "dlr_edan_shared_control_lerobot",
        "cmu_stretch_lerobot",
        "austin_buds_dataset_lerobot",
        "nyu_franka_play_dataset_lerobot",
        "berkeley_cable_routing_lerobot",
        "berkeley_fanuc_manipulation_lerobot",
        "viola_lerobot",
        "jaco_play_lerobot",
        "berkeley_autolab_ur5_lerobot",
        "iamlab_cmu_pickup_insert_lerobot",
        "roboturk_lerobot",
        "taco_play_lerobot",
        "austin_sirius_dataset_lerobot",
        "toto_lerobot",
        "austin_sailor_dataset_lerobot",
        "stanford_hydra_dataset_lerobot",
        "utaustin_mutex_lerobot",
        "fmb_dataset_lerobot",
        "dobbe_lerobot",
        "bridgev2_lerobot",
        "kuka_lerobot",
        "fractal20220817_data_lerobot",
        "furniture_bench_dataset_lerobot",
        "bc_z_lerobot",
        "language_table_lerobot",
        "droid_lerobot",
    ]
    def oxe_process(item: Dict) -> Dict:
        cur_rgb = merge_multiview_rgb(third_rgb=item["cur_third_rgb"], gripper_rgb=item["cur_gripper_rgb"], rgb_size=256)
        goal_rgb = merge_multiview_rgb(third_rgb=item["goal_third_rgb"], gripper_rgb=item["goal_gripper_rgb"], rgb_size=256)
        return extract_tokens(
            item={
                "task_inst": item["task_inst"],
                "robot_states": " ".join(map(str, item["robot_states"].flatten())),
                "action": item["action"],
                "cur_rgb": cur_rgb,
                "goal_rgb": goal_rgb,
            },
            rgb_vq_model=rgb_vq_model,
            action_vq_model=action_vq_model,
        )
    data = {}
    for dataset_name in DATASETS:
        dataset = LeRobotDataset(
            data_dir=os.path.join(data_dir, dataset_name),
            action_chunk_size=action_chunk_size,
            action_flag=action_flag,
            num_chunks=num_chunks,
            chunk_idx=chunk_idx,
            debug=debug,
        )
        if action_flag:
            data[dataset_name] = list(map(lambda x: x["action"], dataset))
        else:
            data[dataset_name] = thread_map(
                oxe_process,
                dataset,
                max_workers=max_workers,
                desc=f"[{chunk_idx}/{num_chunks}] Extract OXE Tokens",
                ncols=100,
            )
            data[dataset_name] = convert_polars(data=data[dataset_name], action_chunk_size=action_chunk_size, num_quantizers=action_vq_model.config.num_quantizers)
    if action_flag:
        data = list(chain(*list(data.values())))
    return data


def decode_video_frames_torchcodec(video_path: str, device: str = "cpu") -> np.ndarray:
    if importlib.util.find_spec("torchcodec"):
        from torchcodec.decoders import VideoDecoder
    else:
        raise ImportError("torchcodec is required but not available.")
    decoder = VideoDecoder(video_path, device=device)
    frames = []
    for frame in decoder:
        frames.append(frame.detach().cpu().numpy())
    video = np.stack(frames)
    return video


def extract_ssv2(
    data_dir: str,
    action_chunk_size: int,
    rgb_vq_model: MagViTv2,
    action_vq_model: ActionRVQModel,
    action_flag: bool,
    num_chunks: int,
    chunk_idx: int,
    max_workers: int,
    debug: bool
) -> None:
    if action_flag:
        return []
    train_data = json.load(open(os.path.join(args.data_dir, "train.json")))
    train_data = get_chunk(train_data, n=num_chunks, k=chunk_idx)
    if debug:
        train_data = train_data[:2]
    def ssv2_process(item: Dict) -> List[Dict]:
        id, desc = item["id"], item["label"]
        video = decode_video_frames_torchcodec(os.path.join(data_dir, "videos", f"{id}.webm"))
        data = []
        for i in range(video.shape[0]):
            j = min(i, max(i + action_chunk_size, video.shape[0] - 1))
            data.append(
                extract_tokens(
                    item={
                        "task_inst": desc,
                        "robot_states": "",
                        "cur_rgb": merge_multiview_rgb(third_rgb=Image.fromarray(np.transpose(video[i], (1, 2, 0))), rgb_size=256),
                        "goal_rgb": merge_multiview_rgb(third_rgb=Image.fromarray(np.transpose(video[j], (1, 2, 0))), rgb_size=256),
                    },
                    rgb_vq_model=rgb_vq_model,
                    action_vq_model=action_vq_model
                )
            )
        return data
    data = thread_map(
        ssv2_process,
        train_data,
        max_workers=max_workers,
        desc=f"[{chunk_idx}/{num_chunks}] Construct SSV2 data",
        ncols=100,
    )
    data = list(chain(*data))
    data = convert_polars(data=data, action_chunk_size=action_chunk_size, num_quantizers=action_vq_model.config.num_quantizers)
    return data


def extract_data(
    dataset: ROBOTDATASET,
    data_dir: str,
    action_chunk_size: int,
    rgb_vq_model: MagViTv2,
    action_vq_model: ActionRVQModel,
    action_flag: bool,
    num_chunks: int,
    chunk_idx: int,
    max_workers: int
) -> List[Dict]:
    dataset_extract_func = {
        ROBOTDATASET.calvin: partial(
            extract_calvin,
            action_chunk_size=action_chunk_size,
            rgb_vq_model=rgb_vq_model,
            action_vq_model=action_vq_model,
            num_chunks=num_chunks,
            chunk_idx=chunk_idx,
            max_workers=max_workers,
            debug=args.debug
        ),
        ROBOTDATASET.libero: partial(
            extract_libero,
            action_chunk_size=action_chunk_size,
            rgb_vq_model=rgb_vq_model,
            action_vq_model=action_vq_model,
            num_chunks=num_chunks,
            chunk_idx=chunk_idx,
            max_workers=max_workers,
            debug=args.debug
        ),
        ROBOTDATASET.oxe: partial(
            extract_oxe,
            action_chunk_size=action_chunk_size,
            rgb_vq_model=rgb_vq_model,
            action_vq_model=action_vq_model,
            num_chunks=num_chunks,
            chunk_idx=chunk_idx,
            max_workers=max_workers,
            debug=args.debug
        ),
        ROBOTDATASET.ssv2: partial(
            extract_ssv2,
            action_chunk_size=action_chunk_size,
            rgb_vq_model=rgb_vq_model,
            action_vq_model=action_vq_model,
            num_chunks=num_chunks,
            chunk_idx=chunk_idx,
            max_workers=max_workers,
            debug=args.debug
        ),
    }
    data = dataset_extract_func[dataset](data_dir=data_dir, action_flag=action_flag)
    return data


def main(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_vq_model = MagViTv2.from_pretrained(args.pretrained_visvq).to(device)
    vision_vq_model.eval()
    freeze_module(vision_vq_model)
    pretrained_actrvq = os.path.join(os.getcwd(), "ckpt", f"ActRVQ_{args.action_chunk_size}steps", args.pretrained_actrvq)
    action_vq_model = ActionRVQModel.from_pretrained(pretrained_actrvq).to(device)
    action_vq_model.eval()
    freeze_module(action_vq_model)

    dataset_dirs = {
        ROBOTDATASET.calvin: args.calvin_data_dir,
        ROBOTDATASET.libero: args.libero_data_dir,
        ROBOTDATASET.oxe: args.oxe_data_dir,
        ROBOTDATASET.ssv2: args.ssv2_data_dir,
    }
    if args.action_flag:
        action_data = []
    for dataset, data_dir in dataset_dirs.items():
        data = extract_data(
            dataset=dataset,
            data_dir=data_dir,
            action_chunk_size=args.action_chunk_size,
            rgb_vq_model=vision_vq_model,
            action_vq_model=action_vq_model,
            action_flag=args.action_flag,
            num_chunks=args.num_chunks,
            chunk_idx=args.chunk_idx,
            max_workers=args.max_workers
        )
        if args.action_flag:
            action_data += data
        else:
            if dataset == ROBOTDATASET.oxe or dataset == ROBOTDATASET.libero:
                for sub_dataset_name, sub_data in data.items():
                    os.makedirs(os.path.join(args.save_dir, sub_dataset_name), exist_ok=True)
                    sub_data.write_parquet(os.path.join(args.save_dir, sub_dataset_name, f"{args.chunk_idx}_{args.num_chunks}.parquet"))
            else:
                os.makedirs(os.path.join(args.save_dir, dataset.name), exist_ok=True)
                data.write_parquet(os.path.join(args.save_dir, dataset.name, f"{args.chunk_idx}_{args.num_chunks}.parquet"))
    if args.action_flag:
        if args.num_chunks == 1:
            os.makedirs(args.save_dir, exist_ok=True)
            np.save(os.path.join(args.save_dir, f"action_{args.action_chunk_size}chunk.npy"), np.array(action_data))
        else:
            os.makedirs(os.path.join(args.save_dir, "action"), exist_ok=True)
            np.save(os.path.join(args.save_dir, "action", f"{args.chunk_idx}_{args.num_chunks}.npy"), np.array(action_data))


if __name__ == "__main__":
    args = get_args()
    main(args=args)