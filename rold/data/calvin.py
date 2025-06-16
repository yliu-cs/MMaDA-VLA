import os
import re
import torch
import autoroot
import numpy as np
from PIL import Image
from random import choice
from typing import List, Dict, Union
from rold.data.utils import image_transform


class CalvinDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        image_path: str,
    ) -> None:
        self.data = np.load(data_path, allow_pickle=True)
        self.image = np.load(image_path, allow_pickle=True).item()
        self.n_steps = int(re.search(r"\d+(?=step)", data_path).group())

    def __len__(self) -> int:
        return len(self.data)
    
    def load_item(self, item: Dict) -> Dict[str, Union[str, torch.Tensor]]:
        task_inst, action = item["desc"], item["action"]
        cur_image, pred_image = self.image[item["cur_image"]], self.image[item["goal_image"]]
        return {
            "task_inst": task_inst,
            "cur_image": cur_image,
            "action_chunk": action,
            "pred_image": pred_image
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        try:
            return self.load_item(self.data[idx])
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return choice(self)
    
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
    data_path = os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN", "training", "calvin_abc_d_8steps.npy")
    image_path = os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN", "training", "calvin_abc_d_8steps_image.npy")
    task = re.search(r'calvin_([a-z_]+)_\d', data_path).group(1)
    dataset = CalvinDataset(
        data_path=data_path,
        image_path=image_path
    )
    print(f"{len(dataset)=}")
    item = dataset[0]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        for k, v in batch.items():
            print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
        break