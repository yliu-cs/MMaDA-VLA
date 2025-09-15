import os
import re
import torch
import autoroot
import polars as pl
from glob import glob
from random import choice
from typing import List, Dict, Union


class RoLDDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths: str) -> None:
        self.data = pl.read_parquet(data_paths)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        item = self.data.row(idx, named=True)
        try:
            return {
                "desc": item["desc"],
                "cur_image": item["cur_image"],
                "pred_image": item["pred_image"],
                "action": item["action"]
            }
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return choice(self)
    
    def collate_fn(self, batch: List[Dict[str, Union[str, torch.Tensor]]]) -> Dict[str, Union[List[str], torch.Tensor]]:
        task_inst = [item["desc"] for item in batch]
        action, cur_image, pred_image = [torch.stack([torch.LongTensor(item[key]) for item in batch]) for key in ("action", "cur_image", "pred_image")]
        return {
            "task_inst": task_inst,
            "cur_image": cur_image,
            "pred_image": pred_image,
            "action": action
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_paths = list(glob(os.path.join(os.sep, "liuyang", "Dataset", "RoLD", "pretrain", "*.parquet")))
    data_paths = [
        os.path.join(os.sep, "liuyang", "Dataset", "RoLD", "pretrain", "bridgev2_lerobot.parquet"),
        os.path.join(os.sep, "liuyang", "Dataset", "RoLD", "pretrain", "calvin_abcd_8steps_pretrain.parquet"),
    ]
    dataset = RoLDDataset(data_paths=data_paths)
    print(f"{len(dataset)=}")
    item = dataset[0]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        for k, v in batch.items():
            print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
        break