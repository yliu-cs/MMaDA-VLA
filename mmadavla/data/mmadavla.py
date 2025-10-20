import os
import torch
import autoroot
import polars as pl
from glob import glob
from random import choice
from typing import List, Dict, Union
from torch.nn.utils.rnn import pad_sequence
from mmadavla.utils.prompt import ignore_id


class MMaDAVLADataset(torch.utils.data.Dataset):
    def __init__(self, data_paths: str) -> None:
        self.data = pl.read_parquet(data_paths)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        item = self.data.row(idx, named=True)
        try:
            return {
                "desc": f"{item['task_inst']}\n{item['robot_states']}",
                "cur_image": item["cur_rgb"],
                "pred_image": item["goal_rgb"],
                "action": item["action"]
            }
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return choice(self)
    
    def collate_fn(self, batch: List[Dict[str, Union[str, torch.Tensor]]]) -> Dict[str, Union[List[str], torch.Tensor]]:
        task_inst = [item["desc"] for item in batch]
        cur_image, pred_image = [torch.stack([torch.LongTensor(item[key]) for item in batch]) for key in ("cur_image", "pred_image")]
        action = pad_sequence([torch.LongTensor(item["action"]) for item in batch], batch_first=True, padding_value=ignore_id)
        return {
            "task_inst": task_inst,
            "cur_image": cur_image,
            "pred_image": pred_image,
            "action": action
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_paths = list(glob(os.path.join(os.sep, "liuyang", "Dataset", "MMaDA-VLA", "vla_5chunk", "*.parquet")))
    dataset = MMaDAVLADataset(data_paths=data_paths)
    print(f"{len(dataset)=}")
    item = dataset[0]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        for k, v in batch.items():
            print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
        break