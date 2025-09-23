import os
import json
import torch
import datasets
import autoroot
import numpy as np
from typing import List, Dict
from datasets import load_dataset
from argparse import ArgumentParser
from mmadavla.utils.misc import get_chunk
from mmadavla.data.utils import (
    load_info,
    NormStats,
    STATS_PATH,
    load_tasks,
    load_stats,
    load_episodes,
    aggregate_stats,
    normalize_action,
    load_episodes_stats,
    decode_video_frames,
    EPISODES_STATS_PATH,
    hf_transform_to_torch,
    get_episode_data_index,
    binarize_gripper_actions,
    get_hf_features_from_features,
)


class LeRobotDatasetMetadata:
    def __init__(self, data_dir: str | None = None) -> None:
        self.data_dir = data_dir
        self.load_metadata()
    
    def load_metadata(self) -> None:
        self.info = load_info(self.data_dir)
        self.tasks, self.task_to_task_index = load_tasks(self.data_dir)
        self.episodes = load_episodes(self.data_dir)
        if os.path.exists(os.path.join(self.data_dir, EPISODES_STATS_PATH)):
            self.episodes_stats = load_episodes_stats(self.data_dir)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))
        elif os.path.exists(os.path.join(self.data_dir, STATS_PATH)):
            self.stats = load_stats(self.data_dir)
        else:
            raise NotImplementedError
    
    def get_video_file_path(self, ep_index: int, vid_key: str) -> str:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return fpath
    
    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size
    
    @property
    def data_path(self) -> str:
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def features(self) -> Dict[str, Dict]:
        return self.info["features"]

    @property
    def image_keys(self) -> List[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> List[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> List[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> Dict[str, List | Dict]:
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> Dict:
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        return self.info["chunks_size"]


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str | None = None,
        tolerance_s: float = 1e-4,
        action_chunk_size: int = 8,
        video_backend: str | None = None,
        action_flag: bool = False,
        action_stats: NormStats | None = None,
        image_key: Dict = None,
        num_chunks: int = 1,
        chunk_idx: int = 0,
        debug: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend
        self.action_chunk_size = action_chunk_size
        self.action_flag = action_flag
        self.action_stats = action_stats
        self.image_key = image_key
        self.meta = LeRobotDatasetMetadata(self.data_dir)
        self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, None)
        self.chunk_indices = []
        for ep_idx in range(self.num_episodes):
            ep_start, ep_end = [self.episode_data_index[key][ep_idx] for key in ("from", "to")]
            actions = torch.stack([self.hf_dataset[i]["action"] for i in range(ep_start, ep_end)])
            actions = torch.cat((actions[:, :6], binarize_gripper_actions(actions[:, -1])[:, None]), dim=1)
            for i in range(ep_start, ep_end + 1):
                start_idx, end_idx = i, min(i + self.action_chunk_size + 1, ep_end)
                action = actions[start_idx - ep_start:end_idx - ep_start]
                if action.shape[0] > self.action_chunk_size:
                    action = action[:self.action_chunk_size, :]
                elif action.shape[0] < self.action_chunk_size:
                    continue
                self.chunk_indices.append({
                    "ep_idx": ep_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "action": action,
                })
        if num_chunks > 1:
            self.chunk_indices = get_chunk(self.chunk_indices, num_chunks, chunk_idx)
            if debug:
                self.chunk_indices = self.chunk_indices[:2]
    
    def load_hf_dataset(self) -> datasets.Dataset:
        hf_dataset = load_dataset("parquet", data_dir=os.path.join(self.data_dir, "data"), split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset
    
    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: Dict[str, List[int]] | None = None,
    ) -> Dict[str, List[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps
    
    def _query_videos(self, query_timestamps: Dict[str, List[float]], ep_idx: int) -> Dict[str, torch.Tensor]:
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = os.path.join(self.data_dir, self.meta.get_video_file_path(ep_idx, vid_key))
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)
        return item

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        return self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)
    
    def __len__(self) -> int:
        return len(self.chunk_indices)
    
    def load_rgb(self, ep_idx: int, current_ts: int) -> Dict:
        query_timestamps = self._get_query_timestamps(current_ts, None)
        video_frames = self._query_videos(query_timestamps, ep_idx)
        return video_frames

    def __getitem__(self, idx: int) -> Dict:
        chunk_info = self.chunk_indices[idx]
        ep_idx, start_idx, end_idx, action = [chunk_info[key] for key in ("ep_idx", "start_idx", "end_idx", "action")]
        chunk_data = []
        for i in range(start_idx, end_idx + 1):
            chunk_data.append(self.hf_dataset[i])

        # for key in self.meta.camera_keys:
        #     key_image = self.load_rgb(ep_idx, chunk_data[0]["timestamp"].item())[key]
        #     key_image = (torch.permute(key_image, (1, 2, 0)).detach().cpu().numpy() * 255).astype(np.uint8)
        #     from PIL import Image
        #     key_image = Image.fromarray(key_image)
        #     key_image.save(os.path.join(os.getcwd(), "media", f"{os.path.basename(self.data_dir)}_{key}.jpg"))
        
        result = {}
        if not self.action_flag:
            result["cur_third_rgb"] = self.load_rgb(ep_idx, chunk_data[0]["timestamp"].item())[self.image_key["third_rgb"]]
            result["cur_gripper_rgb"] = None if self.image_key["gripper_rgb"] is None else self.load_rgb(ep_idx, chunk_data[0]["timestamp"].item())[self.image_key["gripper_rgb"]]
            result["goal_third_rgb"] = self.load_rgb(ep_idx, chunk_data[-1]["timestamp"].item())[self.image_key["third_rgb"]]
            result["goal_gripper_rgb"] = None if self.image_key["gripper_rgb"] is None else self.load_rgb(ep_idx, chunk_data[-1]["timestamp"].item())[self.image_key["gripper_rgb"]]
        for key in ("cur_third_rgb", "cur_gripper_rgb", "goal_third_rgb", "goal_gripper_rgb"):
            if isinstance(result[key], torch.Tensor):
                result[key] = (torch.permute(result[key], (1, 2, 0)).detach().cpu().numpy() * 255).astype(np.uint8)
        result["action"] = action.cpu().numpy()
        # while result["action"].shape[0] < self.action_chunk_size:
        #     add_act = np.zeros_like(result["action"][-1], dtype=result["action"].dtype)
        #     add_act[-1] = result["action"][-1][-1]
        #     result["action"] = np.concatenate([result["action"], add_act[None, :]], axis=0)
        if self.action_stats is not None:
            action = normalize_action(action=result["action"], action_stats=self.action_stats)
        if not self.action_flag:
            result["robot_states"] = chunk_data[0]["observation.state"]
            if "task_index" in chunk_data[0]:
                task_idx = chunk_data[0]["task_index"].item()
                result["task_inst"] = self.meta.tasks[task_idx]
        return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "OpenX-LeRobot"))
    args = parser.parse_args()
    for dataset_name in sorted(os.listdir(args.data_dir)):
        if not dataset_name.endswith("_lerobot") or dataset_name not in ["droid_lerobot", "fmb_dataset_lerobot"]:
            continue
        print(f"{'=' * 20} Loading dataset: {dataset_name} {'=' * 20}")
        try:
            dataset = LeRobotDataset(data_dir=os.path.join(args.data_dir, dataset_name))
        except Exception as e:
            print(f"Error loading dataset: {dataset_name}")
            print(f"{e=}")
            import traceback
            traceback.print_exc()
            continue
        dataset[0]
        # print(f"Total number of episodes: {dataset.meta.total_episodes}, Average number of frames per episode: {dataset.meta.total_frames / dataset.meta.total_episodes:.3f}, Frames per second used during data collection: {dataset.meta.fps}, Robot type: {dataset.meta.robot_type}, keys to access images from cameras: {dataset.meta.camera_keys=}, Features: {dataset.meta.features.keys()}")
        # print(f"Tasks: {dataset.meta.tasks}")

        # print(f"Camera Keys: {dataset.meta.camera_keys=}")
        # print(f"Number of episodes selected: {dataset.num_episodes} Number of frames selected: {dataset.num_frames}")

        # for k, v in dataset[0].items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {v.shape} {v.dtype}")
        #     else:
        #         print(f"{k}: {type(v)}")
        # print(f"{dataset[0]['action']=}")
        # dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=32, shuffle=True)
        # for batch in dataloader:
        #     print(f"{batch['action'].flatten().min().item()=} {batch['action'].flatten().max().item()=}")
        #     break
        # break