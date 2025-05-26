import os
import re
import torch
import numpy as np


class CalvinDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, data_path: str) -> None:
        self.data_dir = data_dir
        self.data = np.load(data_path, allow_pickle=True)
        self.n_steps = int(re.search(r"\d+(?=step)", data_path).group())
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        task_inst, data_chunk = self.data[idx]["desc"], []
        for filename in self.data[idx]["filenames"]:
            episode_data = np.load(os.path.join(self.data_dir, filename))
            data_chunk.append({key: episode_data[key] for key in ["rgb_static", "rel_actions"]})  # rgb_gripper
            break
        action_chunk = [chunk_data["rel_actions"] for chunk_data in data_chunk]
        while len(action_chunk) < self.n_steps:
            add_act = np.zeros_like(action_chunk[-1], dtype=action_chunk[-1].dtype)
            add_act[-1] = action_chunk[-1][-1]
            action_chunk.append(add_act)
        action_chunk = np.stack(action_chunk, axis=0)
        return {
            "task_inst": task_inst,
            "cur_image": data_chunk[0]["rgb_static"],
            "action_chunk": action_chunk,
            "pred_image": data_chunk[-1]["rgb_static"]
        }
    
    def collate_fn(self, batch):
        pass


if __name__ == "__main__":
    data_dir = os.path.join(os.sep, "liuyang", "Dataset", "CALVIN")
    data_path = os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN", "calvin_abc_d_8steps.npy")
    dataset = CalvinDataset(
        data_dir=os.path.join(data_dir, "task_" + (re.search(r'calvin_([a-z_]+)_\d', data_path)).group(1).upper(), "training"),
        data_path=data_path
    )
    print(f"{len(dataset)=}")
    item = dataset[0]
    for k, v in item.items():
        print(f"{k}: {v if isinstance(v, str) else v.shape}")