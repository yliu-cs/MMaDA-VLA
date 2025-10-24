import os
import torch
import numpy as np
from random import sample
from typing import List, Union
import torch.nn.functional as F


class ActionTokenizer(object):
    def __init__(self, num_bins: int = 256, min_val: float = -1., max_val: float = 1., action_chunk_size: int = 5, action_dim: int = 7) -> None:
        self.num_bins, self.min_val, self.max_val = num_bins, min_val, max_val
        self.bins = np.linspace(self.min_val, self.max_val, self.num_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_chunk_size, self.action_dim = action_chunk_size, action_dim
    
    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        action = np.clip(action, a_min=self.min_val, a_max=self.max_val)
        discretized_action = np.digitize(action, self.bins) - 1
        action_ids = discretized_action.reshape(-1, self.action_chunk_size * self.action_dim)
        return action_ids
    
    def decode(self, action_ids: np.ndarray) -> np.ndarray:
        discretized_actions = np.clip(action_ids, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]
    
    @property
    def vocab_size(self) -> int:
        return self.num_bins


if __name__ == "__main__":
    data_path = os.path.join(os.sep, "liuyang", "Dataset", "MMaDA-VLA", "normed_action_5chunk", "libero.npy")
    actions = np.load(data_path)
    act = actions[sample(list(range(actions.shape[0])), 4)]
    action_tokenizer = ActionTokenizer()
    print(f"{action_tokenizer.vocab_size=}")
    act_ids = action_tokenizer(act)
    recon_act = action_tokenizer.decode(act_ids.reshape(-1, action_tokenizer.action_chunk_size, action_tokenizer.action_dim))
    recon_act[..., -1] = np.sign(recon_act[..., -1])
    mse_loss = F.mse_loss(torch.from_numpy(act), torch.from_numpy(recon_act))
    print(f"{mse_loss=}")