import os
import torch
import autoroot
import numpy as np
from PIL import Image
from typing import Tuple
from functools import partial
from rold.utils.misc import quiet
from rold.models.magvitv2 import MagViTv2
from rold.data.utils import image_transform
from tqdm.contrib.concurrent import thread_map
from argparse import Namespace, ArgumentParser


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN", "training"))
    parser.add_argument("--pretrained_visvq", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2"))
    parser.add_argument("--task", type=str, default="ABCD_D")
    parser.add_argument("--max_workers", type=int, default=80)
    return parser.parse_args()


def extract_vis(data: Tuple[str, np.ndarray], vision_vq_model: MagViTv2) -> Tuple[str, np.ndarray]:
    filename, image = data
    image = image_transform(Image.fromarray(image))
    image = image.unsqueeze(0).to(vision_vq_model.device)
    image_token = vision_vq_model.get_code(image)
    image_token = image_token.squeeze().detach().cpu().numpy().tolist()
    return (filename, image_token)


def main(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vision_vq_model = MagViTv2.from_pretrained(args.pretrained_visvq).to(device)
    vision_vq_model.eval()
    vision_vq_model.requires_grad_(False)
    
    data_path = os.path.join(args.data_dir, f"calvin_{args.task.lower()}_image.npy")
    data = np.load(data_path, allow_pickle=True).item()
    data = list(data.items())

    results = thread_map(
        partial(extract_vis, vision_vq_model=vision_vq_model),
        data,
        max_workers=args.max_workers,
        desc="Extracting Vision Codebook",
        ncols=100
    )
    np.save(os.path.join(args.data_dir, f"calvin_{args.task.lower()}_image_token.npy"), dict(results))


if __name__ == "__main__":
    quiet()
    args = parse_args()
    main(args)