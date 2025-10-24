import os
import json
import torch
import logging
import datasets
import importlib
import jsonlines
import torchvision
import numpy as np
from PIL import Image
from itertools import accumulate
from dataclasses import dataclass
from typing import List, Dict, Any
from torchvision import transforms


# ================================================== MMaDA MagViTv2 ==================================================


def image_transform(image: Image.Image, resolution: int = 256, normalize: bool = True) -> torch.Tensor:
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image


# ================================================== LeRobot ==================================================


INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes.jsonl"
STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "/") -> Dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict, sep: str = "/") -> Dict:
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def hf_transform_to_torch(items_dict: Dict):
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, Image.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is None:
            pass
        else:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict


def get_hf_features_from_features(features: Dict) -> datasets.Features:
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"]))
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")
    return datasets.Features(hf_features)


def load_json(fpath: str) -> Any:
    with open(fpath) as f:
        return json.load(f)


def load_jsonlines(fpath: str) -> List[Any]:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def cast_stats_to_numpy(stats: Dict) -> Dict[str, Dict[str, np.ndarray]]:
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_info(local_dir: str) -> Dict:
    info = load_json(os.path.join(local_dir, INFO_PATH))
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info


def load_tasks(local_dir: str) -> tuple[Dict, Dict]:
    tasks = load_jsonlines(os.path.join(local_dir, TASKS_PATH))
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index


def load_episodes(local_dir: str) -> Dict:
    episodes = load_jsonlines(os.path.join(local_dir, EPISODES_PATH))
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}


def load_stats(local_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    if not os.path.exists(os.path.join(local_dir, STATS_PATH)):
        return None
    stats = load_json(os.path.join(local_dir, STATS_PATH))
    return cast_stats_to_numpy(stats)


def load_episodes_stats(local_dir: str) -> Dict:
    episodes_stats = load_jsonlines(os.path.join(local_dir, EPISODES_STATS_PATH))
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }


def get_episode_data_index(
    episode_dicts: Dict[int, Dict], episodes: List[int] | None = None
) -> Dict[str, torch.Tensor]:
    episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in episode_dicts.items()}
    if episodes is not None:
        episode_lengths = {ep_idx: episode_lengths[ep_idx] for ep_idx in episodes}
    cumulative_lengths = list(accumulate(episode_lengths.values()))
    return {
        "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
        "to": torch.LongTensor(cumulative_lengths),
    }


def _assert_type_and_shape(stats_list: List[Dict[str, Dict]]):
    for i in range(len(stats_list)):
        for fkey in stats_list[i]:
            for k, v in stats_list[i][fkey].items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                    )
                if v.ndim == 0:
                    raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
                if k == "count" and v.shape != (1,):
                    raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
                if "image" in fkey and k != "count" and v.shape != (3, 1, 1):
                    raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")


def aggregate_feature_stats(stats_ft_list: List[Dict[str, Dict]]) -> Dict[str, Dict[str, np.ndarray]]:
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count
    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(stats_list: List[Dict[str, Dict]]) -> Dict[str, Dict[str, np.ndarray]]:
    _assert_type_and_shape(stats_list)
    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}
    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)
    return aggregated_stats


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning("'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder")
        return "pyav"


def decode_video_frames_torchcodec(
    video_path: str,
    timestamps: List[float],
    tolerance_s: float,
    device: str = "cpu",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    if importlib.util.find_spec("torchcodec"):
        from torchcodec.decoders import VideoDecoder
    else:
        raise ImportError("torchcodec is required but not available.")
    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    loaded_frames = []
    loaded_ts = []
    metadata = decoder.metadata
    average_fps = metadata.average_fps
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    frames_batch = decoder.get_frames_at(indices=frame_indices)
    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logging.info(f"Frame loaded at timestamp={pts:.4f}")
    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)
    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
    )
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]
    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")
    closest_frames = closest_frames.type(torch.float32) / 255
    assert len(timestamps) == len(closest_frames)
    return closest_frames


def decode_video_frames_torchvision(
    video_path: str,
    timestamps: List[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True
    reader = torchvision.io.VideoReader(video_path, "video")
    first_ts = min(timestamps)
    last_ts = max(timestamps)
    reader.seek(first_ts, keyframes_only=keyframes_only)
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break
    if backend == "pyav":
        reader.container.close()
    reader = None
    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)
    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]
    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")
    closest_frames = closest_frames.type(torch.float32) / 255
    assert len(timestamps) == len(closest_frames)
    return closest_frames


def decode_video_frames(
    video_path: str,
    timestamps: List[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


# ================================================== Open-X Embodiment ==================================================


def binarize_gripper_actions(gripper_actions: torch.Tensor) -> torch.Tensor:
    open_mask = gripper_actions > 0.95
    closed_mask = gripper_actions < 0.05
    in_between_mask = ~(open_mask | closed_mask)
    is_open_float = open_mask.float()
    new_actions = torch.empty_like(gripper_actions)
    carry = gripper_actions[-1].item()
    for i in range(len(gripper_actions)-1, -1, -1):
        if in_between_mask[i]:
            new_actions[i] = carry
        else:
            carry = is_open_float[i].item()
            new_actions[i] = carry
    return new_actions


# ================================================== Normalization ==================================================


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


class RunningStats(object):
    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000
    
    def update(self, batch: np.ndarray) -> None:
        batch = batch.reshape(-1, batch.shape[-1])
        num_ele, vec_len = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch ** 2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vec_len)]
            self._bin_edges = [np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1) for i in range(vec_len)]
        else:
            if vec_len != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()
        self._count += num_ele
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch ** 2, axis=0)
        self._mean += (batch_mean - self._mean) * (num_ele / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_ele / self._count)
        self._update_histograms(batch)
    
    def get_statistics(self) -> NormStats:
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        variance = self._mean_of_squares - self._mean ** 2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)
    
    def _adjust_histograms(self) -> None:
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges
    
    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist
    
    def _compute_quantiles(self, quantiles) -> List[np.ndarray]:
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


def normalize_action(action: np.ndarray, action_stats: NormStats, eps: float = 1e-8) -> np.ndarray:
    normalized = 2 * (action - action_stats.q01) / (action_stats.q99 - action_stats.q01 + eps) - 1
    return np.clip(normalized, -1, 1)


def unnormalize_action(action: np.ndarray, action_stats: NormStats, eps: float = 1e-8) -> np.ndarray:
    action = 0.5 * (action + 1) * (action_stats.q99 - action_stats.q01 + eps) + action_stats.q01
    return action