import os
import re
import torch
import autoroot
import numpy as np
from PIL import Image
from typing import List, Dict, Union
from rold.utils.prompt import Prompting
from rold.data.utils import image_transform


class CalvinDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        data_path: str,
    ) -> None:
        self.data_dir = data_dir
        self.data = np.load(data_path, allow_pickle=True)
        self.n_steps = int(re.search(r"\d+(?=step)", data_path).group())

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, Union[str, torch.Tensor]]:
        task_inst, data_chunk = self.data[idx]["desc"], []
        for filename in self.data[idx]["filenames"]:
            episode_data = np.load(os.path.join(self.data_dir, filename))
            data_chunk.append({key: episode_data[key] for key in ["rgb_static", "rel_actions"]})  # rgb_gripper
        action_chunk = [chunk_data["rel_actions"] for chunk_data in (data_chunk[:-1] if len(data_chunk) >= self.n_steps + 1 else data_chunk)]
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
    
    def collate_fn(self, batch: List[Dict[str, Union[str, torch.Tensor]]]) -> Dict[str, Union[List[str], torch.Tensor]]:
        task_inst = [item["task_inst"] for item in batch]
        action_chunk = torch.stack([torch.from_numpy(item["action_chunk"]) for item in batch]).to(dtype=torch.float)
        cur_image, pred_image = [torch.stack([image_transform(Image.fromarray(item[key])) for item in batch]) for key in ("cur_image", "pred_image")]
        return {
            "task_inst": task_inst,
            "cur_image": cur_image,
            "pred_image": pred_image,
            "action": action_chunk
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(os.sep, "liuyang", "Dataset", "CALVIN")
    data_path = os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN", "calvin_abc_d_8steps.npy")
    task = re.search(r'calvin_([a-z_]+)_\d', data_path).group(1)
    dataset = CalvinDataset(
        data_dir=os.path.join(data_dir, f"task_{task.upper()}", "training"),
        data_path=data_path
    )
    print(f"{len(dataset)=}")
    item = dataset[0]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        for k, v in batch.items():
            print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
        break