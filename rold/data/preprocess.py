import os
import autoroot
import numpy as np
from time import sleep
from tqdm.auto import tqdm
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
    parser.add_argument("--mode", type=str, default="training")
    parser.add_argument("--store_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN"))
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--action", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=20)
    return parser.parse_args()


def preprocess_calvin_data(args: Namespace, indx: List[Tuple[int, int]], ann: List[str]) -> None:
    def get_action_chunk(t: Tuple[int, int]) -> np.ndarray:
        s, e, action_chunk = t[0], t[1], []
        for i in range(args.n_steps):
            if s + i < e + 1:
                action_chunk.append(np.load(os.path.join(args.data_dir, f"episode_{str(s + i).zfill(7)}.npz"))["rel_actions"])
        while len(action_chunk) < args.n_steps:
            add_act = np.zeros_like(action_chunk[-1], dtype=action_chunk[-1].dtype)
            add_act[-1] = action_chunk[-1][-1]
            action_chunk.append(add_act)
        return np.stack(action_chunk)
    def preprocess(t: Tuple[int, int], desc: str) -> np.ndarray:
        s, e = t
        if not all([os.path.exists(os.path.join(args.data_dir, f"episode_{str(i).zfill(7)}.npz")) for i in range(s, e + 1)]):
            return []
        data = []
        for i in range(s, e + 1):
            data.append({
                "desc": desc,
                "action": get_action_chunk((i, min(i + args.n_steps, e))),
                "cur_image": f"episode_{str(i).zfill(7)}.npz",
                "goal_image": f"episode_{str(min(i + args.n_steps, e)).zfill(7)}.npz",
                "filenames": [f"episode_{str(j).zfill(7)}.npz" for j in range(i, min(i + args.n_steps + 1, e + 1))]
            })
        return data
    if not args.merge:
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
            os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_{args.n_steps}steps_{args.chunk_idx}.npy"),
            np.array(list(chain(*data)), dtype=object)
        )
    else:
        if all([os.path.exists(os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_{args.n_steps}steps_{i}.npy")) for i in range(args.num_chunks)]):
            sleep(5)
            data = []
            for i in range(args.num_chunks):
                data += np.load(os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_{args.n_steps}steps_{i}.npy"), allow_pickle=True).tolist()
            np.save(
                os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_{args.n_steps}steps.npy"),
                np.array(data, dtype=object)
            )
            for i in range(args.num_chunks):
                os.remove(os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_{args.n_steps}steps_{i}.npy"))


def preprocess_calvin_image(args: Namespace) -> None:
    if not args.merge:
        calvin_image_data = {}
        episode_filenames = list(filter(lambda x: x.startswith("episode_") and x.endswith(".npz"), os.listdir(args.data_dir)))
        episode_filenames = get_chunk(episode_filenames, n=args.num_chunks, k=args.chunk_idx)
        for episode_filename in tqdm(episode_filenames, desc=f"Preprocess CALVIN \'{args.task.lower().replace('_', ' -> ')}\' Image", ncols=100):
            episode_data = np.load(os.path.join(args.data_dir, episode_filename))
            calvin_image_data[episode_filename] = episode_data["rgb_static"]
        np.save(os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_image_{args.chunk_idx}.npy"), calvin_image_data)
    else:
        if all([os.path.exists(os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_image_{i}.npy")) for i in range(args.num_chunks)]):
            sleep(5)
            calvin_image_data = {}
            for i in range(args.num_chunks):
                calvin_image_data.update(np.load(os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_image_{i}.npy"), allow_pickle=True).item())
            np.save(os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_image.npy"), calvin_image_data)
            for i in range(args.num_chunks):
                os.remove(os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_image_{i}.npy"))


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
    if not args.merge:
        data = thread_map(
            factorize_calvin_data,
            indx,
            desc=f"Preprocess CALVIN \'{args.task.lower().replace('_', ' -> ')}\' action Data",
            ncols=100,
            total=len(indx)
        )
        np.save(
            os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_{args.n_steps}steps_action_{args.chunk_idx}.npy"),
            np.array(list(chain(*data)))
        )
    else:
        if all([os.path.exists(os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_{args.n_steps}steps_action_{i}.npy")) for i in range(args.num_chunks)]):
            sleep(5)
            data = []
            for i in range(args.num_chunks):
                data += np.load(os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_{args.n_steps}steps_action_{i}.npy")).tolist()
            np.save(
                os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_{args.n_steps}steps_action.npy"),
                np.array(data)
            )
            for i in range(args.num_chunks):
                os.remove(os.path.join(args.store_dir, args.mode, f"calvin_{args.task.lower()}_{args.n_steps}steps_action_{i}.npy"))


def main(args: Namespace) -> Dict[str, Union[str, List[str]]]:
    args.data_dir = os.path.join(args.data_dir, f"task_{args.task.upper()}", args.mode)
    language_ann_data = np.load(os.path.join(args.data_dir, "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True).item()
    indx, ann = language_ann_data["info"]["indx"], language_ann_data["language"]["ann"]
    indx, ann = [get_chunk(x, n=args.num_chunks, k=args.chunk_idx) for x in (indx, ann)]
    if not os.path.exists(os.path.join(args.store_dir, args.mode)):
        os.mkdir(os.path.join(args.store_dir, args.mode))
    if not args.action:
        if not args.image:
            preprocess_calvin_data(args, indx, ann)
        else:
            preprocess_calvin_image(args)
    else:
        preprocess_calvin_action(args, indx)


if __name__ == "__main__":
    args = get_args()
    main(args)