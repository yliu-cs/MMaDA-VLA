import os
import json
import torch
import autoroot
import importlib
import numpy as np
from PIL import Image
from typing import Dict
from itertools import chain
from functools import partial
from rold.utils.misc import get_chunk
from rold.models.magvitv2 import MagViTv2
from rold.data.utils import image_transform
from tqdm.contrib.concurrent import thread_map
from argparse import ArgumentParser, Namespace


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "something-something-v2"))
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "SSV2"))
    parser.add_argument("--pretrained_visvq", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2"))
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=150)
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    return args


def decode_video_frames_torchcodec(video_path: str, device: str = "cpu") -> np.ndarray:
    if importlib.util.find_spec("torchcodec"):
        from torchcodec.decoders import VideoDecoder
    else:
        raise ImportError("torchcodec is required but not available.")
    decoder = VideoDecoder(video_path, device=device)
    frames = []
    for frame in decoder:
        frames.append(frame.detach().cpu().numpy())
    video = np.stack(frames)
    return video


def process(item: Dict, vision_vq_model: MagViTv2, action_chunk_size: int) -> Dict:
    id, desc = item["id"], item["label"]
    video = decode_video_frames_torchcodec(os.path.join(args.data_dir, 'videos', f'{id}.webm'))
    data = []
    for i in range(video.shape[0]):
        j = min(i, max(i + action_chunk_size, video.shape[0] - 1))
        cur_image, pred_image = video[i], video[j]
        cur_image = image_transform(Image.fromarray(np.transpose(cur_image, (1, 2, 0))))
        cur_image = cur_image.unsqueeze(0).to(vision_vq_model.device)
        cur_image_token = vision_vq_model.get_code(cur_image)
        cur_image_token = cur_image_token.squeeze().detach().cpu().numpy().astype(np.int16)
        pred_image = image_transform(Image.fromarray(np.transpose(pred_image, (1, 2, 0))))
        pred_image = pred_image.unsqueeze(0).to(vision_vq_model.device)
        pred_image_token = vision_vq_model.get_code(pred_image)
        pred_image_token = pred_image_token.squeeze().detach().cpu().numpy().astype(np.int16)
        data.append({
            "desc": desc,
            "cur_image": cur_image_token,
            "pred_image": pred_image_token,
        })
    return data


def main(args: Namespace) -> None:
    if args.merge:
        data = []
        for chunk_idx in range(args.num_chunks):
            data += np.load(os.path.join(args.save_dir, f"ssv2_{chunk_idx}_{args.num_chunks}.npy"), allow_pickle=True).tolist()
        np.save(os.path.join(args.save_dir, "ssv2.npy"), data)
        for chunk_idx in range(args.num_chunks):
            os.remove(os.path.join(args.save_dir, f"ssv2_{chunk_idx}_{args.num_chunks}.npy"))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_vq_model = MagViTv2.from_pretrained(args.pretrained_visvq).to(device)
    vision_vq_model.eval()
    vision_vq_model.requires_grad_(False)

    train_data = json.load(open(os.path.join(args.data_dir, "train.json")))
    train_data = get_chunk(train_data, args.num_chunks, args.chunk_idx)
    data = thread_map(
        partial(process, vision_vq_model=vision_vq_model, action_chunk_size=args.action_chunk_size),
        train_data,
        max_workers=args.max_workers,
        desc=f"[{args.chunk_idx}/{args.num_chunks}] Processing SSV2",
    )
    data = list(chain(*data))
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, f"ssv2_{args.chunk_idx}_{args.num_chunks}.npy"), data)


if __name__ == "__main__":
    args = get_args()
    main(args)