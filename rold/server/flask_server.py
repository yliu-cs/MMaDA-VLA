import os
import json
import torch
import autoroot
import numpy as np
from PIL import Image
from typing import List
from argparse import ArgumentParser
from transformers import AutoTokenizer
from rold.utils.prompt import Prompting
from rold.models.rold import RoLDModelLM
from flask import Flask, jsonify, request
from rold.models.magvitv2 import MagViTv2
from rold.data.utils import image_transform
from rold.models.actrvq import ActionRVQModel
from rold.utils.diffusion import cosine_mask_schedule


class VLAServer(object):
    def __init__(
        self,
        vision_vq_model: MagViTv2,
        action_vq_model: ActionRVQModel,
        tokenizer: AutoTokenizer,
        rold: RoLDModelLM,
        prompt: Prompting,
        device: torch.device,
        temperature: float,
        timesteps: int
    ) -> None:
        self.vision_vq_model = vision_vq_model
        self.action_vq_model = action_vq_model
        self.tokenizer = tokenizer
        self.rold = rold
        self.prompt = prompt
        self.device = device
        self.temperature = temperature
        self.timesteps = timesteps
        self.mask_token_id = self.rold.config.mask_token_id
        self.mask_schedule = cosine_mask_schedule
    
    def generate_action(
        self,
        task_inst: str,
        image: np.ndarray
    ) -> List[float]:
        cur_image = Image.fromarray(image).convert("RGB")
        cur_image = image_transform(cur_image).unsqueeze(0).to(self.device)
        cur_image_tokens = self.vision_vq_model.get_code(cur_image)
        vision_tokens = torch.ones((1, self.rold.config.vision_num_vq_tokens), dtype=torch.long, device=self.device) * self.mask_token_id
        action_tokens = torch.ones((1, self.rold.config.action_num_vq_tokens), dtype=torch.long, device=self.device) * self.mask_token_id
        input_ids, attention_mask = self.prompt(
            task_inst=[task_inst],
            cur_image_tokens=cur_image_tokens,
            pred_image_tokens=vision_tokens,
            action_tokens=action_tokens,
            pred_image_labels=None,
            action_labels=None
        )
        with torch.no_grad():
            gen_action_ids, gen_vision_ids = self.rold.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                noise_schedule=self.mask_schedule,
                temperature=self.temperature,
                timesteps=self.timesteps,
                vision_seq_len=self.rold.config.vision_num_vq_tokens,
                action_seq_len=self.rold.config.action_num_vq_tokens,
                prompt=self.prompt
            )
        gen_action_ids = torch.clamp(gen_action_ids, max=self.rold.config.action_codebook_size - 1, min=0)
        actions = self.action_vq_model.detokenize(gen_action_ids)
        actions = actions.squeeze(0).cpu().numpy()
        actions = np.clip(actions, a_min=-1., a_max=1.)
        actions = actions.tolist()
        for i in range(len(actions)):
            actions[i][-1] = 1. if actions[i][-1] >= 0 else -1.
        return actions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rold_path", type=str, default=os.path.join(os.getcwd(), "ckpt", "RoLD", "620e690f0e7f7b7ba833357f72eb1807"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()
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
    vla_server = VLAServer(
        vision_vq_model=vision_vq_model,
        action_vq_model=action_vq_model,
        tokenizer=tokenizer,
        rold=rold,
        prompt=prompt,
        device=device,
        temperature=args.temperature,
        timesteps=args.timesteps
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