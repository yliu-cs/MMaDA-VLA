import os
import numpy as np
from tqdm.contrib.concurrent import thread_map
from argparse import ArgumentParser, Namespace


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "CALVIN", "training"))
    args = parser.parse_args()
    return args


def main(args: Namespace) -> None:
    raw_data = np.load(os.path.join(args.data_dir, "calvin_abc_d_8steps_token.npy"), allow_pickle=True).tolist()
    image_map = np.load(os.path.join(args.data_dir, "calvin_abc_d_image_token.npy"), allow_pickle=True).item()
    data = thread_map(
        lambda item: {
            "desc": f"{item['desc']}\n{item['robot_obs']}",
            "cur_image": np.array(image_map[item["cur_image"]], dtype=np.int16),
            "pred_image": np.array(image_map[item["goal_image"]], dtype=np.int16),
            "action": np.array(item["action"], dtype=np.int16)
        },
        raw_data,
        max_workers=100,
        desc="Processing Calvin Pretrain"
    )
    os.makedirs(args.data_dir, exist_ok=True)
    np.save(os.path.join(args.data_dir, "calvin_abc_8steps_pretrain.npy"), data)


if __name__ == "__main__":
    args = get_args()
    main(args)