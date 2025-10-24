import os
import json
import torch
import autoroot
import numpy as np
from PIL import Image
from dataclasses import asdict
from typing import Tuple, Union
from transformers import AutoTokenizer
from mmadavla.utils.prompt import Prompting
from mmadavla.models.magvitv2 import MagViTv2
from mmadavla.models.mmadavla import MMaDAVLAModelLM
from mmadavla.utils.diffusion import cosine_mask_schedule
from mmadavla.models.action_tokenizer import ActionTokenizer
from mmadavla.data.utils import image_transform, unnormalize_action
from mmadavla.data.preprocess import merge_multiview_rgb, load_action_stats
from mmadavla.utils.dllm_cache import dLLMCacheConfig, dLLMCache, register_cache_MMaDA


class MMaDA_VLA_Server(object):
    def __init__(
        self,
        action_stats_path: str = os.path.join(os.sep, "liuyang", "Dataset", "MMaDA-VLA"),
        magvit_path: str = os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2"),
        mmadavla_path: str = os.path.join(os.getcwd(), "ckpt", "MMaDA-VLA", "5afa0d69f888d1335985da8dbab36404"),
        benchmark: str = "libero",  # ["calvin", "libero"]
        device: torch.device = torch.device("cuda"),
        temperature: float = 1.0,
        timesteps: int = 24,
        cache: bool = True,
        prompt_interval_steps: int = 6,
        gen_interval_steps: int = 6,
        transfer_ratio: float = 0.0,
    ) -> None:
        self.device = device
        self.benchmark = benchmark
        self.training_args = json.load(open(os.path.join(mmadavla_path, "args.json")))
        self.action_stats = load_action_stats(os.path.join(action_stats_path, f"action_stats_{self.training_args['action_chunk_size']}chunk.json"))[self.benchmark]
        self.vision_vq_model = MagViTv2.from_pretrained(magvit_path).to(self.device).eval()
        self.vision_vq_model.requires_grad_(False)
        self.action_tokenizer = ActionTokenizer(action_chunk_size=self.training_args["action_chunk_size"])
        self.mmadavla = MMaDAVLAModelLM.from_pretrained(mmadavla_path, torch_dtype=torch.bfloat16).to(self.device).eval()
        self.mmadavla.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.training_args["pretrained_mmada"], padding_side="left")
        self.cache = cache
        if self.cache:
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                transfer_ratio=transfer_ratio
            )))
            register_cache_MMaDA(self.mmadavla, "model.transformer.blocks")
        self.prompt = Prompting(
            tokenizer=self.tokenizer,
            mask_id=self.mmadavla.config.mask_token_id,
            max_text_len=self.training_args["max_text_len"],
            vision_codebook_size=self.mmadavla.config.vision_codebook_size,
            action_codebook_size=self.mmadavla.config.action_codebook_size
        )
        self.mask_token_id = self.mmadavla.config.mask_token_id
        self.mask_schedule = cosine_mask_schedule
        self.temperature = temperature
        self.timesteps = timesteps
    
    def inference(
        self,
        task_inst: str,
        image: Union[np.ndarray, Image.Image],
        gripper_image: Union[np.ndarray, Image.Image],
        merged_image: Union[Image.Image, None] = None,
    ) -> Tuple[Image.Image, torch.Tensor]:
        if merged_image is None:
            image, gripper_image = np.array(image) if isinstance(image, Image.Image) else image, np.array(gripper_image) if isinstance(gripper_image, Image.Image) else gripper_image
            merged_image = merge_multiview_rgb(third_rgb=image, gripper_rgb=gripper_image)
        merged_image_tokens = self.vision_vq_model.get_code(image_transform(merged_image).unsqueeze(0).to(self.device)) + len(self.prompt.tokenizer)
        vision_tokens = torch.ones((1, self.mmadavla.config.vision_num_vq_tokens), dtype=torch.long, device=self.device) * self.mask_token_id
        action_tokens = torch.ones((1, self.mmadavla.config.action_num_vq_tokens), dtype=torch.long, device=self.device) * self.mask_token_id
        input_ids, attention_bias = self.prompt(
            task_inst=[task_inst],
            cur_image_tokens=merged_image_tokens,
            pred_image_tokens=vision_tokens,
            action_tokens=action_tokens,
        )
        if self.cache:
            cache_instance = dLLMCache()
            cache_instance.reset_cache(prompt_length=input_ids.shape[1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action_sampled_ids_list, action_masking_list, vision_sampled_ids_list, vision_masking_list = self.mmadavla.generate(
                input_ids=input_ids,
                attention_bias=attention_bias,
                temperature=self.temperature,
                timesteps=self.timesteps,
                prompt=self.prompt
            )
        images, actions = [], []
        for act, img in zip(action_sampled_ids_list, vision_sampled_ids_list):
            action_ids = torch.clamp(act, max=self.mmadavla.config.action_codebook_size - 1, min=0)
            action_ids = action_ids.detach().cpu().numpy()
            action = self.action_tokenizer.decode(action_ids.reshape(-1, self.action_tokenizer.action_chunk_size, self.action_tokenizer.action_dim))
            action = unnormalize_action(action=action, action_stats=self.action_stats)
            if self.benchmark == "calvin":
                pass
            elif self.benchmark == "libero":
                action[..., -1] = 0 - action[..., -1]
            action = action.squeeze(axis=0)
            action[..., -1] = np.sign(action[..., -1])
            image_ids = torch.clamp(img, max=self.mmadavla.config.vision_codebook_size - 1, min=0)
            image = self.vision_vq_model.decode_code(image_ids)
            image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
            image *= 255.0
            image = Image.fromarray(image.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0])
            images.append(image)
            actions.append(action)
        return images, actions