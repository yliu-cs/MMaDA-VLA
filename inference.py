import os
import json
import torch
import shutil
import autoroot
import numpy as np
from PIL import Image
from rich import print
from random import choice
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
    parser.add_argument("--data_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN", "training"))
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "CALVIN", "task_ABC_D", "training"))
    parser.add_argument("--rold_path", type=str, default=os.path.join(os.getcwd(), "ckpt", "RoLD", "ca1d09542fde601afad882bfb4e2fdff"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=24)
    return parser.parse_args()


def main(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    training_args = json.load(open(os.path.join(args.rold_path, "args.json")))
    vision_vq_model = MagViTv2.from_pretrained(training_args["pretrained_visvq"]).to(device).eval()
    vision_vq_model.requires_grad_(False)
    action_vq_model = ActionRVQModel.from_pretrained(
        os.path.join(
            os.getcwd(),
            "ckpt",
            f"ActRVQ_{training_args['task'].lower()}_{training_args['action_chunk_size']}steps",
            training_args["actrvq"]
        )
    ).to(device).eval()
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
    mask_schedule = cosine_mask_schedule

    data = np.load(os.path.join(args.data_path, "calvin_abc_d_8steps.npy"), allow_pickle=True)
    data = choice(data)
    task_inst, filenames = data["desc"], data["filenames"]
    cur_image = Image.fromarray(np.load(os.path.join(args.data_dir, filenames[0]))["rgb_static"]).convert("RGB")
    cur_image.save(os.path.join(os.getcwd(), "demo", "cur_image.png"))
    cur_image = image_transform(cur_image).unsqueeze(0).to(device)
    cur_image_tokens = vision_vq_model.get_code(cur_image) + len(prompt.tokenizer)
    goal_image = Image.fromarray(np.load(os.path.join(args.data_dir, filenames[-1]))["rgb_static"]).convert("RGB")
    goal_image_tokens = vision_vq_model.get_code(image_transform(goal_image).unsqueeze(0).to(device))
    goal_image.save(os.path.join(os.getcwd(), "demo", "goal_image.png"))

    gt_actions = np.array([np.load(os.path.join(args.data_dir, filename))["rel_actions"] for filename in filenames])
    if gt_actions.shape[0] == 9:
        gt_actions = gt_actions[:-1]
    while gt_actions.shape[0] < 8:
        add_act = np.zeros_like(gt_actions[-1], dtype=gt_actions[-1].dtype)
        add_act[-1] = gt_actions[-1][-1]
        gt_actions = np.concatenate([gt_actions, add_act[None, :]], axis=0)
    gt_actions = torch.from_numpy(gt_actions).unsqueeze(0).to(dtype=torch.float, device=device)
    gt_action_ids = action_vq_model.tokenize(gt_actions)
    gt_action_ids_list = gt_action_ids.flatten().detach().cpu().numpy().tolist()
    print(" " * 8 + " ".join(list(map(lambda x: f"{str(x):>3}✨", gt_action_ids_list))))

    vision_tokens = torch.ones((1, rold.config.vision_num_vq_tokens), dtype=torch.long, device=device) * mask_token_id
    action_tokens = torch.ones((1, rold.config.action_num_vq_tokens), dtype=torch.long, device=device) * mask_token_id

    print(f"{'Input':<8}" + " ".join(list(map(lambda x: f"{'msk' if x == mask_token_id else str(x):>3}📍", torch.where(action_tokens == mask_token_id, action_tokens, action_tokens - rold.config.llm_vocab_size - rold.config.vision_codebook_size).flatten().detach().cpu().numpy().tolist()))))

    input_ids, attention_mask = prompt(
        task_inst=[task_inst],
        cur_image_tokens=cur_image_tokens,
        pred_image_tokens=vision_tokens,
        action_tokens=action_tokens,
        pred_image_labels=None,
        action_labels=None
    )

    with torch.no_grad():
        gen_action_ids_list, gen_action_masking_list, gen_vision_ids_list, gen_vision_masking_list = rold.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            noise_schedule=mask_schedule,
            temperature=args.temperature,
            timesteps=args.timesteps,
            prompt=prompt
        )
    
    vision_store_dir = os.path.join(os.getcwd(), "demo", "pred_image")
    if os.path.exists(vision_store_dir):
        shutil.rmtree(vision_store_dir)
    os.makedirs(vision_store_dir)

    for step, (step_action_ids, step_action_masking, step_vision_ids, step_vision_masking) in enumerate(zip(gen_action_ids_list, gen_action_masking_list, gen_vision_ids_list, gen_vision_masking_list)):
        step_vision_ids = torch.clamp(step_vision_ids, max=rold.config.vision_codebook_size - 1, min=0)
        images = vision_vq_model.decode_code(step_vision_ids)
        # print(f"{step_action_ids.shape=} {step_action_ids.flatten().min().item()=} {step_action_ids.flatten().max().item()=}")
        step_action_ids = torch.clamp(step_action_ids, max=rold.config.action_codebook_size - 1, min=0)
        actions = action_vq_model.detokenize(step_action_ids)

        step_action_ids_list, print_id_list = step_action_ids.flatten().detach().cpu().numpy().tolist(), []
        step_action_masking_list = step_action_masking.flatten().detach().cpu().numpy().tolist()
        for gt_action_id, step_action_id, step_action_masking_ele in zip(gt_action_ids_list, step_action_ids_list, step_action_masking_list):
            print_id_list.append(f"[green]{str(step_action_id):>3}[/green]" if gt_action_id == step_action_id else f"[red]{str(step_action_id):>3}[/red]")
            print_id_list[-1] += f"{'🧊' if not step_action_masking_ele else '🔥'}"
        acc = sum(1 for gt_action_id, step_action_id in zip(gt_action_ids_list, step_action_ids_list) if gt_action_id == step_action_id) / len(gt_action_ids_list)
        
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        save_image = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        save_image = Image.fromarray(save_image[0])
        save_image.save(os.path.join(vision_store_dir, f"{step + 1}.png"))

        mse = float(F.mse_loss(actions, gt_actions))
        print(f"[{str(step + 1):>2}/{args.timesteps}] {' '.join(print_id_list)} [orange]{acc=:.2f}[/orange] {mse=:.4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)