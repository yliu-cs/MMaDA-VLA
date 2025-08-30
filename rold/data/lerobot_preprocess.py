import os
import autoroot
import numpy as np
from typing import List
from tqdm.auto import tqdm
from itertools import chain
from rold.data.lerobot import LeRobotDataset
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "OpenX-LeRobot"))
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--store_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "OpenX"))
    parser.add_argument("--action", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--max_workers", type=int, default=1)
    return parser.parse_args()


def process_action(args: Namespace) -> None:
    def process_dataset(dataset_name: str) -> List[np.ndarray]:
        data_dir = os.path.join(args.data_dir, dataset_name)
        dataset = LeRobotDataset(data_dir=data_dir)
        return [item["action"] for item in tqdm(dataset, desc=f"Processing {dataset_name}")]
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        actions = list(executor.map(process_dataset, list(sorted(os.listdir(args.data_dir)))))
    actions = np.stack(list(chain(*actions)), axis=0)
    os.makedirs(args.store_dir, exist_ok=True)
    np.save(os.path.join(args.store_dir, f"openx_{args.n_steps}steps_action.npy"), actions)


def main(args: Namespace) -> None:
    if args.action:
        process_action(args)


if __name__ == "__main__":
    args = get_args()
    main(args)