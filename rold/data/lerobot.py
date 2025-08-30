import os
import json
import torch
import datasets
import autoroot
from typing import List, Dict
from datasets import load_dataset
from argparse import ArgumentParser
from rold.utils.misc import get_chunk
from rold.data.utils import (
    load_info,
    STATS_PATH,
    load_tasks,
    load_stats,
    load_episodes,
    aggregate_stats,
    load_episodes_stats,
    decode_video_frames,
    EPISODES_STATS_PATH,
    hf_transform_to_torch,
    get_episode_data_index,
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
        num_chunks: int = 1,
        chunk_idx: int = 0,
    ) -> None:
        self.data_dir = data_dir
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend
        self.action_chunk_size = action_chunk_size
        self.meta = LeRobotDatasetMetadata(self.data_dir)
        self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, None)
        self.chunk_indices = []
        for ep_idx in range(self.num_episodes):
            ep_start, ep_end = [self.episode_data_index[key][ep_idx] for key in ("from", "to")]
            for i in range(ep_start, ep_end + 1):
                if i != min(i + self.action_chunk_size + 1, ep_end):
                    self.chunk_indices.append({
                        "ep_idx": ep_idx,
                        "start_idx": i,
                        "end_idx": min(i + self.action_chunk_size + 1, ep_end)
                    })
        if num_chunks > 1:
            self.chunk_indices = get_chunk(self.chunk_indices, num_chunks, chunk_idx)
    
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
    
    def load_image(self, ep_idx: int, current_ts: int) -> Dict:
        query_timestamps = self._get_query_timestamps(current_ts, None)
        video_frames = self._query_videos(query_timestamps, ep_idx)
        return video_frames

    def __getitem__(self, idx: int) -> Dict:
        chunk_info = self.chunk_indices[idx]
        ep_idx = chunk_info["ep_idx"]
        start_idx = chunk_info["start_idx"]
        end_idx = chunk_info["end_idx"]
        
        chunk_data = []
        for i in range(start_idx, end_idx + 1):
            item = self.hf_dataset[i]
            chunk_data.append(item)
        
        result = {}
        result["cur_image"] = self.load_image(ep_idx, chunk_data[0]["timestamp"].item())[self.meta.camera_keys[0]]
        result["pred_image"] = self.load_image(ep_idx, chunk_data[-1]["timestamp"].item())[self.meta.camera_keys[0]]
        result["action"] = []
        for i in range(min(len(chunk_data) - 1, self.action_chunk_size)):
            result["action"].append(chunk_data[i]["action"])
        while len(result["action"]) < self.action_chunk_size:
            add_act = torch.zeros_like(result["action"][-1], dtype=result["action"][-1].dtype, device=result["action"][-1].device)
            add_act[-1] = result["action"][-1][-1]
            result["action"].append(add_act)
        result["action"] = torch.stack(result["action"])

        result["robot_obs"] = chunk_data[0]["observation.state"]
        if "task_index" in chunk_data[0]:
            task_idx = chunk_data[0]["task_index"].item()
            result["task"] = self.meta.tasks[task_idx]
        return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "OpenX-LeRobot"))
    args = parser.parse_args()
    for dataset_name in sorted(os.listdir(args.data_dir)):
        if not dataset_name.endswith("_lerobot") and dataset_name != "viola_lerobot":
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
        print(f"Total number of episodes: {dataset.meta.total_episodes}, Average number of frames per episode: {dataset.meta.total_frames / dataset.meta.total_episodes:.3f}, Frames per second used during data collection: {dataset.meta.fps}, Robot type: {dataset.meta.robot_type}, keys to access images from cameras: {dataset.meta.camera_keys=}, Features: {dataset.meta.features.keys()}")
        print(f"Tasks: {dataset.meta.tasks}")

        print(f"Number of episodes selected: {dataset.num_episodes} Number of frames selected: {dataset.num_frames}")

        dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=32, shuffle=True)
        for batch in dataloader:
            print(f"{batch['cur_image'].shape=} {batch['pred_image'].shape=} {batch['action'].shape=}")
            # print(f"{batch['task']=}")
            break
            
        break