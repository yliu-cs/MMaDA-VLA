import os
import torch
import autoroot
import numpy as np
from itertools import chain
from typing import Dict, Any
from functools import partial
from rold.utils.misc import quiet, get_chunk
from rold.models.actrvq import ActionRVQModel
from tqdm.contrib.concurrent import thread_map
from argparse import Namespace, ArgumentParser


actrvq_map = {
    "abc_d_5": "e82c437a83bfa9ec0ac4e6b048c05e74",
    "abc_d_8": "36a391f3d2e0d405d7d39f100571a139",
    "abcd_d_8": "3597a1acdea9602d3e3575e17f74a7b6",
}


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "CALVIN"))
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN", "training"))
    parser.add_argument("--pretrained_visvq", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2"))
    parser.add_argument("--task", type=str, default="ABCD_D")
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=100)
    return parser.parse_args()


def extract_act(data: Dict[str, Any], action_vq_model: ActionRVQModel, raw_data_dir: str) -> Dict[str, Any]:
    action = torch.from_numpy(data["action"]).to(dtype=torch.float, device=action_vq_model.device)
    action = action_vq_model.tokenize(action.unsqueeze(0))
    data["action"] = action.squeeze(0).detach().cpu().numpy().tolist()

    # add robot_obs to instruction
    robot_obs = np.load(os.path.join(raw_data_dir, data["cur_image"]))["robot_obs"].flatten()
    data["robot_obs"] = " ".join(map(str, robot_obs))
    return data


def main(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    actrvq_version = actrvq_map[f"{args.task.lower()}_{args.action_chunk_size}"]
    pretrained_actrvq = os.path.join(os.getcwd(), "ckpt", f"ActRVQ_{args.task.lower()}_{args.action_chunk_size}steps", actrvq_version)
    action_vq_model = ActionRVQModel.from_pretrained(pretrained_actrvq).to(device)
    action_vq_model.eval()
    action_vq_model.requires_grad_(False)
    
    data_path = os.path.join(args.data_dir, f"calvin_{args.task.lower()}_{args.action_chunk_size}steps.npy")
    data = np.load(data_path, allow_pickle=True)

    results = thread_map(
        partial(extract_act, action_vq_model=action_vq_model, raw_data_dir=os.path.join(args.raw_data_dir, f"task_{args.task.upper()}", "training")),
        data,
        max_workers=args.max_workers,
        desc="Extracting Action Codebook",
        ncols=100
    )
    np.save(os.path.join(args.data_dir, f"calvin_{args.task.lower()}_{args.action_chunk_size}steps_token.npy"), results)


if __name__ == "__main__":
    quiet()
    args = parse_args()
    main(args)