import os
import torch
import autoroot
import numpy as np
import transformers
from glob import glob
from transformers import Trainer
from dataclasses import dataclass, field, asdict
from rold.utils.misc import quiet, str_datetime, hash_str
from rold.models.actrvq import ActionRVQConfig, ActionRVQModel


@dataclass
class ModelArguments:
    dims: list[int] = field(default_factory=lambda: [7, 2048, 2048, 2048, 512])
    levels: list[int] = field(default_factory=lambda: [8, 5, 5, 3])
    num_quantizers: int = field(default=4)


@dataclass
class DataArguments:
    task: str = field(default="ABC_D")
    n_steps: int = field(default=8)
    data_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN", "training"))


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pass


class ActionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str) -> None:
        self.data = np.load(data_path)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return {"action": torch.from_numpy(self.data[index]).float()}


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = f"{training_args.output_dir}_{data_args.task.lower()}_{data_args.n_steps}steps"
    training_args.output_dir = os.path.join(training_args.output_dir, hash_str(" ".join(list(map(str, [model_args, data_args, training_args]))) + str_datetime()))
    if training_args.local_rank == 0:
        print(f"{training_args.output_dir=}")
    act_rvq = ActionRVQModel(ActionRVQConfig(**asdict(model_args)))
    dataset = ActionDataset(data_path=os.path.join(data_args.data_path, f"calvin_{data_args.task.lower()}_{data_args.n_steps}steps_action.npy"))
    trainer = Trainer(model=act_rvq, args=training_args, train_dataset=dataset)
    trainer.train(resume_from_checkpoint=True if list(glob(os.path.join(training_args.output_dir, "checkpoint-*"))) else False)
    trainer.save_state()
    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    quiet()
    main()