import os
import json
import torch
import autoroot
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer
from rold.utils.prompt import Prompting
from rold.models.rold import RoLDModelLM
from rold.models.magvitv2 import MagViTv2
from rold.data.utils import image_transform
from rold.models.actrvq import ActionRVQModel
from argparse import ArgumentParser, Namespace
from rold.utils.diffusion import cosine_mask_schedule


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--rold_path", type=str, default=os.path.join(os.getcwd(), "ckpt", "RoLD", "475663168de5c2b71cd2beb439f691a6"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=24)
    return parser.parse_args()


def main(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    training_args = json.load(open(os.path.join(args.rold_path, "args.json")))
    vision_vq_model = MagViTv2.from_pretrained(training_args["pretrained_visvq"]).to(device).eval()
    vision_vq_model.requires_grad_(False)
    action_vq_model = ActionRVQModel.from_pretrained(training_args["pretrained_actrvq"]).to(device).eval()
    action_vq_model.requires_grad_(False)
    rold = RoLDModelLM.from_pretrained(args.rold_path, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(training_args["pretrained_mmada"], padding_side="left")

    prompt = Prompting(
        tokenizer=tokenizer,
        max_text_len=training_args["max_text_len"],
        vision_codebook_size=rold.config.vision_codebook_size,
        action_codebook_size=rold.config.action_codebook_size
    )

    mask_token_id = rold.config.mask_token_id
    vision_tokens = torch.ones((1, rold.config.vision_num_vq_tokens), dtype=torch.long, device=device) * mask_token_id
    action_tokens = torch.ones((1, rold.config.action_num_vq_tokens), dtype=torch.long, device=device) * mask_token_id
    mask_schedule = cosine_mask_schedule

    task_inst = [json.load(open(os.path.join(os.getcwd(), "demo", "demo.json")))["desc"]]
    cur_image = Image.open(os.path.join(os.getcwd(), "demo", "cur_image.png")).convert("RGB")
    cur_image = image_transform(cur_image).unsqueeze(0).to(device)
    cur_image_tokens = vision_vq_model.get_code(cur_image)

    input_ids, attention_mask = prompt(
        task_inst=task_inst,
        cur_image_tokens=cur_image_tokens,
        pred_image_tokens=vision_tokens,
        action_tokens=action_tokens,
        pred_image_labels=None,
        action_labels=None
    )

    with torch.no_grad():
        gen_action_ids, gen_vision_ids = rold.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            noise_schedule=mask_schedule,
            temperature=args.temperature,
            timesteps=args.timesteps,
            vision_seq_len=rold.config.vision_num_vq_tokens,
            action_seq_len=rold.config.action_num_vq_tokens,
            prompt=prompt
        )
    
    gen_vision_ids = torch.clamp(gen_vision_ids, max=rold.config.vision_codebook_size - 1, min=0)
    images = vision_vq_model.decode_code(gen_vision_ids)
    gen_action_ids = torch.clamp(gen_action_ids, max=rold.config.action_codebook_size - 1, min=0)
    actions = action_vq_model.detokenize(gen_action_ids)
    
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    save_image = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    save_image = Image.fromarray(save_image[0])
    save_image.save(os.path.join(os.getcwd(), "demo", "pred_image.png"))

    gt_action = np.load(os.path.join(os.getcwd(), "demo", "gt_action.npy"))

    print(f"{float(F.mse_loss(actions, torch.from_numpy(gt_action).unsqueeze(0).to(actions.device)))=}")


if __name__ == "__main__":
    args = get_args()
    main(args)