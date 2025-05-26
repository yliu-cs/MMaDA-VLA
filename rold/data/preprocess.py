import os
import autoroot
import numpy as np
from time import sleep
from itertools import chain
from rold.utils.misc import get_chunk
from typing import List, Tuple, Dict, Union
from argparse import ArgumentParser, Namespace
from tqdm.contrib.concurrent import thread_map


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "CALVIN"))
    parser.add_argument("--task", type=str, default="ABC_D")
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--store_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN"))
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=4)
    return parser.parse_args()


def preprocess_calvin_data(args: Namespace, indx: List[Tuple[int, int]], ann: List[str]) -> None:
    def preprocess(t: Tuple[int, int], desc: str) -> np.ndarray:
        s, e = t
        if not all([os.path.exists(os.path.join(args.data_dir, f"episode_{str(i).zfill(7)}.npz")) for i in range(s, e + 1)]):
            return []
        data = []
        for i in range(s, e + 1):
            data.append({
                "desc": desc
                , "filenames": [f"episode_{str(j).zfill(7)}.npz" for j in range(i, min(i + args.n_steps + 1, e + 1))]
            })
        return data
    data = thread_map(
        preprocess,
        indx,
        ann,
        max_workers=args.max_workers,
        desc=f"Preprocessing CALVIN \'{args.task.lower().replace('_', ' -> ')}\' data",
        ncols=100,
        total=len(indx)
    )
    np.save(
        os.path.join(args.store_dir, f"calvin_{args.task.lower()}_{args.n_steps}steps_{args.chunk_idx}.npy"),
        np.array(list(chain(*data)), dtype=object)
    )

    if all([os.path.exists(os.path.join(args.store_dir, f"calvin_{args.task.lower()}_{args.n_steps}steps_{i}.npy")) for i in range(args.num_chunks)]):
        sleep(5)
        data = []
        for i in range(args.num_chunks):
            data += np.load(os.path.join(args.store_dir, f"calvin_{args.task.lower()}_{args.n_steps}steps_{i}.npy"), allow_pickle=True).tolist()
        np.save(
            os.path.join(args.store_dir, f"calvin_{args.task.lower()}_{args.n_steps}steps.npy"),
            np.array(data, dtype=object)
        )
        for i in range(args.num_chunks):
            os.remove(os.path.join(args.store_dir, f"calvin_{args.task.lower()}_{args.n_steps}steps_{i}.npy"))

def preprocess_calvin_action(args: Namespace, indx: List[Tuple[int, int]]) -> None:
    def factorize_calvin_data(t: Tuple[int, int]) -> np.ndarray:
        s, e, sub_data = t[0], t[1], []
        for step in range(s, e + 1):
            action_chunk = []
            for i in range(args.n_steps):
                if step + i < e + 1:
                    action_chunk.append(np.load(os.path.join(args.data_dir, f"episode_{str(step + i).zfill(7)}.npz"))["rel_actions"])
            while len(action_chunk) < args.n_steps:
                add_act = np.zeros_like(action_chunk[-1], dtype=action_chunk[-1].dtype)
                add_act[-1] = action_chunk[-1][-1]
                action_chunk.append(add_act)
            sub_data.append(np.stack(action_chunk))
        return sub_data
    data = thread_map(
        factorize_calvin_data, 
        indx,
        desc=f"Preprocess CALVIN \'{args.task.lower().replace('_', ' -> ')}\' action Data",
        ncols=100,
        total=len(indx)
    )
    np.save(
        os.path.join(args.store_dir, f"calvin_{args.task.lower()}_{args.n_steps}steps_action.npy"),
        np.array(list(chain(*data)))
    )

    if all([os.path.exists(os.path.join(args.store_dir, f"calvin_{args.task.lower()}_{args.n_steps}steps_action_{i}.npy")) for i in range(args.num_chunks)]):
        sleep(5)
        data = []
        for i in range(args.num_chunks):
            data += np.load(os.path.join(args.store_dir, f"calvin_{args.task.lower()}_{args.n_steps}steps_action_{i}.npy")).tolist()
        np.save(
            os.path.join(args.store_dir, f"calvin_{args.task.lower()}_{args.n_steps}steps_action.npy"),
            np.array(data)
        )
        for i in range(args.num_chunks):
            os.remove(os.path.join(args.store_dir, f"calvin_{args.task.lower()}_{args.n_steps}steps_action_{i}.npy"))


def main(args: Namespace) -> Dict[str, Union[str, List[str]]]:
    args.data_dir = os.path.join(args.data_dir, f"task_{args.task.upper()}", "training")
    language_ann_data = np.load(os.path.join(args.data_dir, "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True).item()
    indx, ann = language_ann_data["info"]["indx"], language_ann_data["language"]["ann"]
    indx, ann = [get_chunk(x, n=args.num_chunks, k=args.chunk_idx) for x in (indx, ann)]
    preprocess_calvin_data(args, indx, ann)
    preprocess_calvin_action(args, indx)


if __name__ == "__main__":
    args = get_args()
    main(args)