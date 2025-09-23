import os
import torch
import autoroot
import numpy as np
import transformers
from glob import glob
from typing import List
from transformers import Trainer
from dataclasses import dataclass, field, asdict
from mmadavla.utils.misc import quiet, str_datetime, hash_str
from mmadavla.models.actrvq import ActionRVQConfig, ActionRVQModel


@dataclass
class ModelArguments:
    dims: list[int] = field(default_factory=lambda: [7, 2048, 2048, 2048, 512])
    levels: list[int] = field(default_factory=lambda: [8, 5, 5, 3])
    num_quantizers: int = field(default=4)


@dataclass
class DataArguments:
    n_steps: int = field(default=8)
    data_path: str = field(default=os.path.join(os.sep, "liuyang", "Dataset", "MMaDA-VLA"))


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pass


class ActionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: List[str]) -> None:
        self.data = np.concatenate([np.load(file) for file in glob(os.path.join(data_dir, "*.npy"))], axis=0)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return {"action": torch.from_numpy(self.data[index]).float()}


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = f"{training_args.output_dir}_{data_args.n_steps}chunk"
    training_args.output_dir = os.path.join(training_args.output_dir, hash_str(" ".join(list(map(str, [model_args, data_args, training_args]))) + str_datetime()))
    act_rvq = ActionRVQModel(ActionRVQConfig(**asdict(model_args)))
    dataset = ActionDataset(data_dir=os.path.join(data_args.data_path, f"normed_action_{data_args.n_steps}chunk"))
    trainer = Trainer(model=act_rvq, args=training_args, train_dataset=dataset)
    trainer.accelerator.print(f"{training_args.output_dir=}")
    trainer.train(resume_from_checkpoint=True if list(glob(os.path.join(training_args.output_dir, "checkpoint-*"))) else False)
    trainer.save_state()
    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    quiet()
    main()