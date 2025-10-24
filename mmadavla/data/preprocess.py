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
from glob import glob
from tqdm.auto import tqdm
from itertools import chain
from functools import partial
from typing import List, Dict, Tuple, Union
from mmadavla.utils.prompt import ignore_id
from mmadavla.models.magvitv2 import MagViTv2
from tqdm.contrib.concurrent import thread_map
from argparse import ArgumentParser, Namespace
from mmadavla.data.lerobot import LeRobotDataset
from mmadavla.models.action_tokenizer import ActionTokenizer
from mmadavla.utils.misc import quiet, get_chunk, freeze_module
from mmadavla.data.utils import image_transform, RunningStats, NormStats, normalize_action


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
    parser.add_argument("--ssv2_data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "something-something-v2"))
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "MMaDA-VLA"))
    parser.add_argument("--pretrained_visvq", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2"))
    parser.add_argument("--action_chunk_size", type=int, default=5)
    parser.add_argument("--action_flag", action="store_true")
    parser.add_argument("--norm_action", action="store_true")
    parser.add_argument("--merge", action="store_true")
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


def extract_action_token(action: np.ndarray, action_tokenizer: ActionTokenizer) -> np.ndarray:
    action_token = action_tokenizer(action[None, ...])
    action_token = action_token[0].astype(np.int16)
    return action_token


def extract_tokens(item: Dict, rgb_vq_model: MagViTv2, action_tokenizer: ActionTokenizer) -> Dict:
    cur_rgb, goal_rgb = [extract_rgb_token(rgb=item[key], rgb_vq_model=rgb_vq_model) for key in ("cur_rgb", "goal_rgb")]
    if "action" in item:
        action = extract_action_token(action=item["action"], action_tokenizer=action_tokenizer)
    else:
        action = np.array([ignore_id] * (action_tokenizer.action_chunk_size * action_tokenizer.action_dim)).astype(np.int16)
    return {
        "task_inst": item["task_inst"],
        "robot_states": item["robot_states"],
        "action": action,
        "cur_rgb": cur_rgb,
        "goal_rgb": goal_rgb,
    }


def convert_polars(data: Union[List, Dict], action_chunk_size: int) -> pl.DataFrame:
    data = pl.DataFrame(data)
    data = data.with_columns([
        pl.col("cur_rgb").map_elements(lambda x: x.flatten().tolist()).cast(pl.List(pl.Int16)),
        pl.col("goal_rgb").map_elements(lambda x: x.flatten().tolist()).cast(pl.List(pl.Int16)),
    ])
    data = data.with_columns([
        pl.col("action").map_elements(lambda x: x.tolist()).cast(pl.List(pl.Int16)),
    ]) if "action" in data.columns else data.with_columns([
        pl.lit([ignore_id] * action_chunk_size).cast(pl.List(pl.Int16)).alias("action"),
    ])
    return data


def extract_calvin(
    data_dir: str,
    action_chunk_size: int,
    rgb_vq_model: MagViTv2,
    action_tokenizer: ActionTokenizer,
    action_flag: bool,
    action_stats: NormStats,
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
        action_chunk = np.array(action_chunk)
        return action_chunk
    def calvin_process(t: Tuple[int, int], task_inst: str) -> List[Dict]:
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
            while action.shape[0] < action_chunk_size:
                action = np.concatenate((action, np.array([0, 0, 0, 0, 0, 0, action[-1, -1]])[None, :]), axis=0)
            if action_stats is not None:
                action = normalize_action(action=action, action_stats=action_stats)
            if action_flag:
                data.append(action)
            else:
                data.append(
                    extract_tokens(
                        item={
                            "task_inst": task_inst,
                            "robot_states": " ".join(map(str, cur_episode["robot_obs"].flatten())),
                            "action": action,
                            "cur_rgb": cur_rgb,
                            "goal_rgb": goal_rgb,
                        },
                        rgb_vq_model=rgb_vq_model,
                        action_tokenizer=action_tokenizer,
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
        data = convert_polars(data=data, action_chunk_size=action_chunk_size)
    return data


def extract_libero(
    data_dir: str,
    action_chunk_size: int,
    rgb_vq_model: MagViTv2,
    action_tokenizer: ActionTokenizer,
    action_flag: bool,
    action_stats: NormStats,
    num_chunks: int,
    chunk_idx: int,
    max_workers: int,
    debug: bool
) -> List[Dict]:
    # SUITE = ["object"]
    SUITE = ["spatial", "goal", "object", "90", "10"]
    data = {}
    for suite in SUITE:
        data[suite] = []
        for task in tqdm(list(map(lambda x: os.path.splitext(x)[0].replace("_demo", ""), os.listdir(os.path.join(data_dir, f"libero_{suite}_no_noops")))), desc=f"Loading LIBERO {suite} tasks", ncols=100):
            task_data = h5py.File(os.path.join(data_dir, f"libero_{suite}_no_noops", f"{task}_demo.hdf5"), "r")["data"]
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
                        action = np.concatenate((action, np.array([0, 0, 0, 0, 0, 0, action[-1, -1]])[None, :]), axis=0)
                    action[..., -1] = 0 - action[..., -1]
                    if action_stats is not None:
                        action = normalize_action(action=np.array(action), action_stats=action_stats)
                    if not action_flag:
                        cur_rgb = merge_multiview_rgb(third_rgb=third_rgbs[j][::-1, ::-1], gripper_rgb=gripper_rgbs[j][::-1, ::-1], rgb_size=256)
                        goal_rgb = merge_multiview_rgb(third_rgb=third_rgbs[min(k + 1, actions.shape[0] - 1)][::-1, ::-1], gripper_rgb=gripper_rgbs[min(k + 1, actions.shape[0] - 1)][::-1, ::-1], rgb_size=256)
                        item = {
                            "task_inst": task.replace("_", " "),
                            "robot_states": " ".join(map(str, robot_states[j].flatten())),
                            "action": action,
                            "cur_rgb": cur_rgb,
                            "goal_rgb": goal_rgb,
                        }
                    data[suite].append(action if action_flag else item)
    if not action_flag:
        for suite in SUITE:
            data[suite] = thread_map(
                partial(extract_tokens, rgb_vq_model=rgb_vq_model, action_tokenizer=action_tokenizer),
                get_chunk(data[suite], n=num_chunks, k=chunk_idx),
                max_workers=max_workers,
                desc=f"[{chunk_idx}/{num_chunks}] Extract LIBERO {suite} Tokens",
                ncols=100,
            )
            data[suite] = convert_polars(data=data[suite], action_chunk_size=action_chunk_size)
    else:
        data = list(chain(*list(data.values())))   # Merge LIBERO Actions
    return data


def extract_oxe(
    data_dir: str,
    action_chunk_size: int,
    rgb_vq_model: MagViTv2,
    action_tokenizer: ActionTokenizer,
    action_flag: bool,
    action_stats: Dict[str, NormStats],
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
    ImageKey = {
        "austin_buds_dataset_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.wrist_image"},
        "austin_sailor_dataset_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.wrist_image"},
        "austin_sirius_dataset_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.wrist_image"},
        "berkeley_fanuc_manipulation_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.wrist_image"},
        "furniture_bench_dataset_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.wrist_image"},
        "iamlab_cmu_pickup_insert_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.wrist_image"},
        "stanford_hydra_dataset_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.wrist_image"},
        "utaustin_mutex_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.wrist_image"},
        "bc_z_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": None},
        "berkeley_autolab_ur5_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.hand_image"},
        "berkeley_cable_routing_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.wrist225_image"},
        "cmu_stretch_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": None},
        "dlr_edan_shared_control_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": None},
        "fractal20220817_data_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": None},
        "jaco_play_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.image_wrist"},
        "kuka_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": None},
        "nyu_franka_play_dataset_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": "observation.images.image_additional_view"},
        "toto_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": None},
        "ucsd_kitchen_dataset_lerobot": {"third_rgb": "observation.images.image", "gripper_rgb": None},
        "bridgev2_lerobot": {"third_rgb": "observation.images.image_0", "gripper_rgb": "observation.images.image_1"},
        "taco_play_lerobot": {"third_rgb": "observation.images.rgb_static", "gripper_rgb": "observation.images.rgb_gripper"},
        "viola_lerobot": {"third_rgb": "observation.images.agentview_rgb", "gripper_rgb": "observation.images.eye_in_hand_rgb"},
        "language_table_lerobot": {"third_rgb": "observation.images.rgb", "gripper_rgb": None},
        "roboturk_lerobot": {"third_rgb": "observation.images.front_rgb", "gripper_rgb": None},
        "fmb_dataset_lerobot": {"third_rgb": "observation.images.image_side_1.jpg", "gripper_rgb": "observation.images.image_wrist_1.jpg"},
        "droid_lerobot": {"third_rgb": "observation.images.exterior_image_1_left", "gripper_rgb": "observation.images.wrist_image_left.jpg"},
    }
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
            action_tokenizer=action_tokenizer,
        )
    data = {}
    for dataset_name in DATASETS:
        dataset = LeRobotDataset(
            data_dir=os.path.join(data_dir, dataset_name),
            action_chunk_size=action_chunk_size,
            action_flag=action_flag,
            action_stats=action_stats[dataset_name],
            image_key=ImageKey[dataset_name],
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
            data[dataset_name] = convert_polars(data=data[dataset_name], action_chunk_size=action_chunk_size)
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
    action_tokenizer: ActionTokenizer,
    action_flag: bool,
    action_stats: Union[NormStats, None],
    num_chunks: int,
    chunk_idx: int,
    max_workers: int,
    debug: bool
) -> None:
    if action_flag:
        return []
    train_data = json.load(open(os.path.join(data_dir, "train.json")))
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
                        "cur_rgb": merge_multiview_rgb(third_rgb=np.transpose(video[i], (1, 2, 0)), rgb_size=256),
                        "goal_rgb": merge_multiview_rgb(third_rgb=np.transpose(video[j], (1, 2, 0)), rgb_size=256),
                    },
                    rgb_vq_model=rgb_vq_model,
                    action_tokenizer=action_tokenizer,
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
    data = convert_polars(data=data, action_chunk_size=action_chunk_size)
    return data


def extract_data(
    dataset: ROBOTDATASET,
    data_dir: str,
    action_chunk_size: int,
    rgb_vq_model: MagViTv2,
    action_tokenizer: ActionTokenizer,
    action_flag: bool,
    action_stats: Union[Dict, None],
    num_chunks: int,
    chunk_idx: int,
    max_workers: int
) -> List[Dict]:
    dataset_extract_func = {
        ROBOTDATASET.calvin: partial(
            extract_calvin,
            action_chunk_size=action_chunk_size,
            rgb_vq_model=rgb_vq_model,
            action_tokenizer=action_tokenizer,
            num_chunks=num_chunks,
            chunk_idx=chunk_idx,
            max_workers=max_workers,
            debug=args.debug
        ),
        ROBOTDATASET.libero: partial(
            extract_libero,
            action_chunk_size=action_chunk_size,
            rgb_vq_model=rgb_vq_model,
            action_tokenizer=action_tokenizer,
            num_chunks=num_chunks,
            chunk_idx=chunk_idx,
            max_workers=max_workers,
            debug=args.debug
        ),
        ROBOTDATASET.oxe: partial(
            extract_oxe,
            action_chunk_size=action_chunk_size,
            rgb_vq_model=rgb_vq_model,
            action_tokenizer=action_tokenizer,
            num_chunks=num_chunks,
            chunk_idx=chunk_idx,
            max_workers=max_workers,
            debug=args.debug
        ),
        ROBOTDATASET.ssv2: partial(
            extract_ssv2,
            action_chunk_size=action_chunk_size,
            rgb_vq_model=rgb_vq_model,
            action_tokenizer=action_tokenizer,
            num_chunks=num_chunks,
            chunk_idx=chunk_idx,
            max_workers=max_workers,
            debug=args.debug
        ),
    }
    if action_stats is None:
        stats = None
    else:
        if dataset == ROBOTDATASET.ssv2:
            stats = None
        elif dataset == ROBOTDATASET.calvin or dataset == ROBOTDATASET.libero:
            stats = action_stats[dataset.name]
        elif dataset == ROBOTDATASET.oxe:
            stats = action_stats
    data = dataset_extract_func[dataset](
        data_dir=data_dir,
        action_flag=action_flag,
        action_stats=stats
    )
    return data


def generate_action_stats(action_file: str) -> None:
    normalizer = RunningStats()
    action = np.load(action_file)
    normalizer.update(action)
    stats = normalizer.get_statistics()
    return {os.path.splitext(os.path.basename(action_file))[0]: stats}


def save_action_stats(action_stats: Dict[str, NormStats], save_path: str) -> None:
    for dataset in action_stats.keys():
        action_stats[dataset] = vars(action_stats[dataset])
        for key in action_stats[dataset]:
            if isinstance(action_stats[dataset][key], np.ndarray):
                action_stats[dataset][key] = action_stats[dataset][key].tolist()
    with open(save_path, "w") as f:
        json.dump(action_stats, f, indent=4)


def load_action_stats(action_stats_path: str) -> Dict[str, NormStats]:
    action_stats = json.load(open(action_stats_path))
    for dataset in action_stats.keys():
        for key in action_stats[dataset]:
            if isinstance(action_stats[dataset][key], list):
                action_stats[dataset][key] = np.array(action_stats[dataset][key])
        action_stats[dataset] = NormStats(**action_stats[dataset])
    return action_stats


def main(args: Namespace) -> None:
    if args.norm_action:
        if not os.path.exists(os.path.join(args.save_dir, f"action_stats_{args.action_chunk_size}chunk.json")):
            action_stats = {}
            for action_file in tqdm(glob(os.path.join(args.save_dir, f"raw_action_{args.action_chunk_size}chunk", "*.npy")), desc="Generating Action Stats", ncols=100):
                sub_action_stats = generate_action_stats(action_file=action_file)
                action_stats.update(sub_action_stats)
            save_action_stats(action_stats=action_stats, save_path=os.path.join(args.save_dir, f"action_stats_{args.action_chunk_size}chunk.json"))
        return

    if args.merge:
        merge_dir = os.path.join(args.save_dir, f"vla_{args.action_chunk_size}chunk")
        merge_folders = list(filter(os.path.isdir, glob(os.path.join(merge_dir, "*"))))
        for merge_folder in tqdm(merge_folders, desc="Merge Folders", ncols=100):
            if os.path.exists(os.path.join(merge_dir, f"{os.path.basename(merge_folder)}.parquet")):
                continue
            data_files = glob(os.path.join(merge_folder, "*.parquet"))
            num_chunks = list(set(list(map(lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[1]), data_files))))
            assert len(num_chunks) == 1
            num_chunks = num_chunks[0]
            assert all([os.path.exists(os.path.join(merge_folder, f"{i}_{num_chunks}.parquet")) for i in range(num_chunks)])
            merged_data = pl.read_parquet([os.path.join(merge_folder, f"{i}_{num_chunks}.parquet") for i in range(num_chunks)])
            merged_data.write_parquet(os.path.join(merge_dir, f"{os.path.basename(merge_folder)}.parquet"))
        return
    
    action_stats_path = os.path.join(args.save_dir, f"action_stats_{args.action_chunk_size}chunk.json")
    action_stats = load_action_stats(action_stats_path=action_stats_path) if os.path.exists(action_stats_path) else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.action_flag:
        vision_vq_model, action_tokenizer = None, None
    else:
        vision_vq_model = MagViTv2.from_pretrained(args.pretrained_visvq).to(device)
        vision_vq_model.eval()
        freeze_module(vision_vq_model)
        action_tokenizer = ActionTokenizer(action_chunk_size=args.action_chunk_size)

    dataset_dirs = {
        # ROBOTDATASET.calvin: args.calvin_data_dir,
        ROBOTDATASET.libero: args.libero_data_dir,
        ROBOTDATASET.ssv2: args.ssv2_data_dir,
        ROBOTDATASET.oxe: args.oxe_data_dir,
    }
    if not args.action_flag:
        args.save_dir = os.path.join(args.save_dir, f"vla_{args.action_chunk_size}chunk")
    for dataset, data_dir in dataset_dirs.items():
        data = extract_data(
            dataset=dataset,
            data_dir=data_dir,
            action_chunk_size=args.action_chunk_size,
            rgb_vq_model=vision_vq_model,
            action_tokenizer=action_tokenizer,
            action_flag=args.action_flag,
            action_stats=action_stats,
            num_chunks=args.num_chunks,
            chunk_idx=args.chunk_idx,
            max_workers=args.max_workers,
        )
        if args.action_flag:
            if dataset == ROBOTDATASET.ssv2:
                continue
            if args.num_chunks == 1:
                if dataset == ROBOTDATASET.oxe:
                    for sub_dataset_name, sub_data in data.items():
                        os.makedirs(os.path.join(args.save_dir, f"raw_action_{args.action_chunk_size}chunk"), exist_ok=True)
                        np.save(os.path.join(args.save_dir, f"raw_action_{args.action_chunk_size}chunk", f"{sub_dataset_name}.npy"), sub_data)
                else:
                    os.makedirs(os.path.join(args.save_dir, f"raw_action_{args.action_chunk_size}chunk"), exist_ok=True)
                    np.save(os.path.join(args.save_dir, f"raw_action_{args.action_chunk_size}chunk", f"{dataset.name}.npy"), data)
            else:
                raise NotImplementedError
        else:
            if dataset == ROBOTDATASET.oxe or dataset == ROBOTDATASET.libero:
                for sub_dataset_name, sub_data in data.items():
                    os.makedirs(os.path.join(args.save_dir, sub_dataset_name), exist_ok=True)
                    sub_data.write_parquet(os.path.join(args.save_dir, sub_dataset_name, f"{args.chunk_idx}_{args.num_chunks}.parquet"))
            else:
                os.makedirs(os.path.join(args.save_dir, dataset.name), exist_ok=True)
                data.write_parquet(os.path.join(args.save_dir, dataset.name, f"{args.chunk_idx}_{args.num_chunks}.parquet"))


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args=args)