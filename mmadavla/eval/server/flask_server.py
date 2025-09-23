import os
import json
import torch
import autoroot
import numpy as np
from PIL import Image
from typing import List
from dataclasses import asdict
from argparse import ArgumentParser
from transformers import AutoTokenizer
from flask import Flask, jsonify, request
from mmadavla.utils.prompt import Prompting
from mmadavla.models.magvitv2 import MagViTv2
from mmadavla.data.utils import image_transform
from mmadavla.models.actrvq import ActionRVQModel
from mmadavla.models.mmadavla import MMaDAVLAModelLM
from mmadavla.utils.diffusion import cosine_mask_schedule
from mmadavla.utils.dllm_cache import dLLMCacheConfig, dLLMCache, register_cache_MMaDA


class VLAServer(object):
    def __init__(
        self,
        vision_vq_model: MagViTv2,
        action_vq_model: ActionRVQModel,
        tokenizer: AutoTokenizer,
        mmadavla: MMaDAVLAModelLM,
        prompt: Prompting,
        device: torch.device,
        temperature: float,
        timesteps: int,
        cache: bool = False,
        prompt_interval_steps: int = 6,
        gen_interval_steps: int = 6,
        transfer_ratio: float = 0.0
    ) -> None:
        self.vision_vq_model = vision_vq_model
        self.action_vq_model = action_vq_model
        self.tokenizer = tokenizer
        self.mmadavla = mmadavla
        self.cache = cache
        if self.cache:
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                transfer_ratio=transfer_ratio
            )))
            register_cache_MMaDA(self.mmadavla, "model.transformer.blocks")
        self.prompt = prompt
        self.device = device
        self.temperature = temperature
        self.timesteps = timesteps
        self.mask_token_id = self.mmadavla.config.mask_token_id
        self.mask_schedule = cosine_mask_schedule
    
    def generate_action(
        self,
        task_inst: str,
        image: np.ndarray
    ) -> List[float]:
        cur_image = Image.fromarray(image).convert("RGB")
        cur_image = image_transform(cur_image).unsqueeze(0).to(self.device)
        cur_image_tokens = self.vision_vq_model.get_code(cur_image) + len(self.prompt.tokenizer)
        vision_tokens = torch.ones((1, self.mmadavla.config.vision_num_vq_tokens), dtype=torch.long, device=self.device) * self.mask_token_id
        action_tokens = torch.ones((1, self.mmadavla.config.action_num_vq_tokens), dtype=torch.long, device=self.device) * self.mask_token_id
        input_ids, attention_mask = self.prompt(
            task_inst=[task_inst],
            cur_image_tokens=cur_image_tokens,
            pred_image_tokens=vision_tokens,
            action_tokens=action_tokens,
            pred_image_labels=None,
            action_labels=None
        )
        if self.cache:
            cache_instance = dLLMCache()
            cache_instance.reset_cache(prompt_length=input_ids.shape[1])
        with torch.no_grad():
            gen_action_ids, _, gen_vision_ids, _ = self.mmadavla.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                noise_schedule=self.mask_schedule,
                temperature=self.temperature,
                timesteps=self.timesteps,
                prompt=self.prompt
            )
        gen_action_ids, gen_vision_ids = gen_action_ids[-1], gen_vision_ids[-1]
        gen_action_ids = torch.clamp(gen_action_ids, max=self.mmadavla.config.action_codebook_size - 1, min=0)
        actions = self.action_vq_model.detokenize(gen_action_ids)
        actions = actions.squeeze(0).cpu().numpy()
        actions = np.clip(actions, a_min=-1., a_max=1.)
        print(f"{task_inst=} {actions[..., -1].tolist()=}")
        actions = actions.tolist()
        for i in range(len(actions)):
            actions[i][-1] = 1. if actions[i][-1] >= 0.5 else -1.
        return actions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mmadavla_path", type=str, default=os.path.join(os.getcwd(), "ckpt", "MMaDA-VLA", "5d0a3d7f8cfea1e615db5bcc180a4eab"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=24)
    parser.add_argument("--port", type=int, default=36657)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_args = json.load(open(os.path.join(args.mmadavla_path, "args.json")))
    task = training_args["task"] if "task" in training_args else ("abcd_d" if "abcd" in "".join(training_args["data_paths"]) else "abc_d").upper()
    if "pretrained_visvq" in training_args:
        vision_vq_model = MagViTv2.from_pretrained(training_args["pretrained_visvq"]).to(device).eval()
    else:
        vision_vq_model = MagViTv2.from_pretrained(os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2")).to(device).eval()
    vision_vq_model.requires_grad_(False)
    action_vq_model = ActionRVQModel.from_pretrained(
        os.path.join(
            os.getcwd(),
            "ckpt",
            f"ActRVQ_{training_args['action_chunk_size']}chunk",
            "ad585dfb97d77ac7651bc5623b3635d6"
        )
    ).to(device).eval()
    action_vq_model.requires_grad_(False)
    mmadavla = MMaDAVLAModelLM.from_pretrained(args.mmadavla_path, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(training_args["pretrained_mmada"], padding_side="left")
    prompt = Prompting(
        tokenizer=tokenizer,
        max_text_len=training_args["max_text_len"],
        vision_codebook_size=mmadavla.config.vision_codebook_size,
        action_codebook_size=mmadavla.config.action_codebook_size
    )
    vla_server = VLAServer(
        vision_vq_model=vision_vq_model,
        action_vq_model=action_vq_model,
        tokenizer=tokenizer,
        mmadavla=mmadavla,
        prompt=prompt,
        device=device,
        temperature=args.temperature,
        timesteps=args.timesteps,
        cache=args.cache,
    )
    flask_app = Flask(__name__)
    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            image = np.frombuffer(request.files["img_static"].read(), dtype=np.uint8).reshape((200, 200, 3))
            content = json.loads(request.files["json"].read())
            task_inst = content["instruction"]
            actions = vla_server.generate_action(task_inst=task_inst, image=image)
            return jsonify(actions)
    flask_app.run(host="0.0.0.0", port=args.port)