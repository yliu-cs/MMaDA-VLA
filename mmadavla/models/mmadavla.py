import os
import torch
import autoroot
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Callable, List
from transformers import PretrainedConfig
from mmadavla.models.mmada import MMaDAModelLM
from mmadavla.utils.prompt import Prompting, ignore_id
from mmadavla.utils.diffusion import cosine_mask_schedule, mask_by_random_topk


class MMaDAVLAConfig(PretrainedConfig):
    model_type = "mmadavla"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            "vision_codebook_size",
            "vision_num_vq_tokens",
            "action_codebook_size",
            "action_num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size",
            "mask_token_id"
        ]
        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])


class MMaDAVLAModelLM(MMaDAModelLM):
    config_class = MMaDAVLAConfig
    base_model_prefix = "model"
    def __init__(self, config: MMaDAVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    def forward_process(
        self,
        input_ids: torch.Tensor,
        attention_bias: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        logits = self(input_ids, attention_bias=attention_bias).logits
        loss = F.cross_entropy(logits.contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1), ignore_index=ignore_id)
        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_bias: torch.Tensor = None,
        temperature: float = 1.0,
        timesteps: int = 18,
        noise_schedule: Callable = cosine_mask_schedule,
        generator: torch.Generator = None,
        mask_token_id: int = 126336,
        prompt: Prompting = None,
        **kwargs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        num_new_special_tokens = 0
        action_input_ids = input_ids[:, -(self.config.action_num_vq_tokens + 1):-1].clone()
        action_input_ids = torch.where(action_input_ids == mask_token_id, mask_token_id, action_input_ids - len(prompt.tokenizer) - prompt.vision_codebook_size - num_new_special_tokens)
        vision_input_ids = input_ids[:, -(self.config.vision_num_vq_tokens + self.config.action_num_vq_tokens + 3):-(self.config.action_num_vq_tokens + 3)].clone()
        vision_input_ids = torch.where(vision_input_ids == mask_token_id, mask_token_id, vision_input_ids - len(prompt.tokenizer) - num_new_special_tokens)

        action_sampled_ids_list, action_masking_list, vision_sampled_ids_list, vision_masking_list = [], [], [], []
        for step in range(timesteps):
            logits = self(input_ids, attention_bias=attention_bias).logits
            action_logits = logits[:, -(self.config.action_num_vq_tokens + 1):-1, len(prompt.tokenizer) + num_new_special_tokens + prompt.vision_codebook_size:len(prompt.tokenizer) + num_new_special_tokens + prompt.vision_codebook_size + prompt.action_codebook_size]
            vision_logits = logits[:, -(self.config.vision_num_vq_tokens + self.config.action_num_vq_tokens + 3):-(self.config.action_num_vq_tokens + 3), len(prompt.tokenizer) + num_new_special_tokens:len(prompt.tokenizer) + num_new_special_tokens + prompt.vision_codebook_size]

            action_probs = action_logits.softmax(dim=-1)
            action_sampled = action_probs.reshape(-1, action_logits.size(-1))
            action_sampled_ids = torch.multinomial(action_sampled, 1, generator=generator)[:, 0].view(*action_probs.shape[:-1])
            action_unknown_map = action_input_ids == mask_token_id
            action_sampled_ids = torch.where(action_unknown_map, action_sampled_ids, action_input_ids)
            vision_probs = vision_logits.softmax(dim=-1)
            vision_sampled = vision_probs.reshape(-1, vision_logits.size(-1))
            vision_sampled_ids = torch.multinomial(vision_sampled, 1, generator=generator)[:, 0].view(*vision_probs.shape[:-1])
            vision_unknown_map = vision_input_ids == mask_token_id
            vision_sampled_ids = torch.where(vision_unknown_map, vision_sampled_ids, vision_input_ids)

            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))

            action_selected_probs = torch.gather(action_probs, -1, action_sampled_ids.long()[..., None])
            action_selected_probs = action_selected_probs.squeeze(-1)
            vision_selected_probs = torch.gather(vision_probs, -1, vision_sampled_ids.long()[..., None])
            vision_selected_probs = vision_selected_probs.squeeze(-1)
            action_selected_probs = torch.where(action_unknown_map, action_selected_probs, torch.finfo(action_selected_probs.dtype).max)
            vision_selected_probs = torch.where(vision_unknown_map, vision_selected_probs, torch.finfo(vision_selected_probs.dtype).max)

            action_mask_len = (self.config.action_num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(action_logits.device)
            action_mask_len = torch.max(
                torch.tensor([1], device=action_logits.device),
                torch.min(action_unknown_map.sum(dim=-1, keepdim=True) - 1, action_mask_len)
            )
            vision_mask_len = (self.config.vision_num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(vision_logits.device)
            vision_mask_len = torch.max(
                torch.tensor([1], device=vision_logits.device),
                torch.min(vision_unknown_map.sum(dim=-1, keepdim=True) - 1, vision_mask_len)
            )

            temperature = temperature * (1.0 - ratio)
            action_masking = mask_by_random_topk(action_mask_len, action_selected_probs, temperature, generator)
            vision_masking = mask_by_random_topk(vision_mask_len, vision_selected_probs, temperature, generator)

            input_ids[:, -(self.config.action_num_vq_tokens + 1):-1] = torch.where(
                action_masking,
                mask_token_id,
                action_sampled_ids + len(prompt.tokenizer) + num_new_special_tokens + prompt.vision_codebook_size
            )
            action_input_ids = torch.where(action_masking, mask_token_id, action_sampled_ids)
            input_ids[:, -(self.config.vision_num_vq_tokens + self.config.action_num_vq_tokens + 3):-(self.config.action_num_vq_tokens + 3)] = torch.where(
                vision_masking,
                mask_token_id,
                vision_sampled_ids + len(prompt.tokenizer) + num_new_special_tokens
            )
            vision_input_ids = torch.where(vision_masking, mask_token_id, vision_sampled_ids)

            action_sampled_ids_list.append(action_sampled_ids)
            action_masking_list.append(action_masking)
            vision_sampled_ids_list.append(vision_sampled_ids)
            vision_masking_list.append(vision_masking)
        return action_sampled_ids_list, action_masking_list, vision_sampled_ids_list, vision_masking_list

    @torch.no_grad()
    def batch_generate(
        self,
        input_ids: torch.Tensor,
        attention_bias: torch.Tensor = None,
        temperature: float = 1.0,
        timesteps: int = 18,
        noise_schedule: Callable = cosine_mask_schedule,
        generator: torch.Generator = None,
        mask_token_id: int = 126336,
        prompt: Prompting = None,
    ) -> List[torch.Tensor]:
        num_new_special_tokens = 0
        action_input_ids = input_ids[:, -(self.config.action_num_vq_tokens + 1):-1].clone()
        action_input_ids = torch.where(action_input_ids == mask_token_id, mask_token_id, action_input_ids - len(prompt.tokenizer) - prompt.vision_codebook_size - num_new_special_tokens)
        vision_input_ids = input_ids[:, -(self.config.vision_num_vq_tokens + self.config.action_num_vq_tokens + 3):-(self.config.action_num_vq_tokens + 3)].clone()
        vision_input_ids = torch.where(vision_input_ids == mask_token_id, mask_token_id, vision_input_ids - len(prompt.tokenizer) - num_new_special_tokens)

        for step in range(timesteps):
            logits = self(input_ids, attention_bias=attention_bias).logits
            action_logits = logits[:, -(self.config.action_num_vq_tokens + 1):-1, len(prompt.tokenizer) + num_new_special_tokens + prompt.vision_codebook_size:len(prompt.tokenizer) + num_new_special_tokens + prompt.vision_codebook_size + prompt.action_codebook_size]
            vision_logits = logits[:, -(self.config.vision_num_vq_tokens + self.config.action_num_vq_tokens + 3):-(self.config.action_num_vq_tokens + 3), len(prompt.tokenizer) + num_new_special_tokens:len(prompt.tokenizer) + num_new_special_tokens + prompt.vision_codebook_size]

            action_probs = action_logits.softmax(dim=-1)
            action_sampled = action_probs.reshape(-1, action_logits.size(-1))
            action_sampled_ids = torch.multinomial(action_sampled, 1, generator=generator)[:, 0].view(*action_probs.shape[:-1])
            action_unknown_map = action_input_ids == mask_token_id
            action_sampled_ids = torch.where(action_unknown_map, action_sampled_ids, action_input_ids)
            vision_probs = vision_logits.softmax(dim=-1)
            vision_sampled = vision_probs.reshape(-1, vision_logits.size(-1))
            vision_sampled_ids = torch.multinomial(vision_sampled, 1, generator=generator)[:, 0].view(*vision_probs.shape[:-1])
            vision_unknown_map = vision_input_ids == mask_token_id
            vision_sampled_ids = torch.where(vision_unknown_map, vision_sampled_ids, vision_input_ids)

            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))

            action_selected_probs = torch.gather(action_probs, -1, action_sampled_ids.long()[..., None])
            action_selected_probs = action_selected_probs.squeeze(-1)
            vision_selected_probs = torch.gather(vision_probs, -1, vision_sampled_ids.long()[..., None])
            vision_selected_probs = vision_selected_probs.squeeze(-1)
            action_selected_probs = torch.where(action_unknown_map, action_selected_probs, torch.finfo(action_selected_probs.dtype).max)
            vision_selected_probs = torch.where(vision_unknown_map, vision_selected_probs, torch.finfo(vision_selected_probs.dtype).max)

            action_mask_len = (self.config.action_num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(action_logits.device)
            action_mask_len = torch.max(
                torch.tensor([1], device=action_logits.device),
                torch.min(action_unknown_map.sum(dim=-1, keepdim=True) - 1, action_mask_len)
            )
            vision_mask_len = (self.config.vision_num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(vision_logits.device)
            vision_mask_len = torch.max(
                torch.tensor([1], device=vision_logits.device),
                torch.min(vision_unknown_map.sum(dim=-1, keepdim=True) - 1, vision_mask_len)
            )

            temperature = temperature * (1.0 - ratio)
            action_masking = mask_by_random_topk(action_mask_len, action_selected_probs, temperature, generator)
            vision_masking = mask_by_random_topk(vision_mask_len, vision_selected_probs, temperature, generator)

            input_ids[:, -(self.config.action_num_vq_tokens + 1):-1] = torch.where(
                action_masking,
                mask_token_id,
                action_sampled_ids + len(prompt.tokenizer) + num_new_special_tokens + prompt.vision_codebook_size
            )
            action_input_ids = torch.where(action_masking, mask_token_id, action_sampled_ids)
            input_ids[:, -(self.config.vision_num_vq_tokens + self.config.action_num_vq_tokens + 3):-(self.config.action_num_vq_tokens + 3)] = torch.where(
                vision_masking,
                mask_token_id,
                vision_sampled_ids + len(prompt.tokenizer) + num_new_special_tokens
            )
            vision_input_ids = torch.where(vision_masking, mask_token_id, vision_sampled_ids)

        return action_sampled_ids