import os
import torch
import autoroot
import numpy as np
import torch.nn.functional as F
from transformers import PretrainedConfig
from mmadavla.models.mmada import MMaDAModelLM
from typing import Tuple, Union, Callable, List
from mmadavla.utils.prompt import Prompting, ignore_id, special_token_mappings
from mmadavla.utils.diffusion import cosine_mask_schedule, get_num_transfer_tokens, add_gumbel_noise


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


def locate_modal_range(sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    soi_token_id, eoi_token_id = special_token_mappings["<|soi|>"], special_token_mappings["<|eoi|>"]
    soa_token_id, eoa_token_id = special_token_mappings["<|soa|>"], special_token_mappings["<|eoa|>"]
    # print(f"{soi_token_id=} {eoi_token_id=} {soa_token_id=} {eoa_token_id=}")
    soi_positions, eoi_positions, soa_positions, eoa_positions = [
        (sequence == key).nonzero(as_tuple=True)[0] for key in (soi_token_id, eoi_token_id, soa_token_id, eoa_token_id)
    ]
    # print(f"{soi_positions=} {eoi_positions=} {soa_positions=} {eoa_positions=}")
    return soi_positions, eoi_positions, soa_positions, eoa_positions


def calc_ti2ia_loss(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[Tuple[List[torch.Tensor], torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
    (batch_size, _), device = labels.shape, labels.device
    vision_logits, vision_losses, action_logits, action_losses = [], [], [], []
    for i in range(batch_size):
        sample_labels, sample_logits = labels[i], logits[i]
        # print(f"{(sample_labels.to(torch.long)).detach().cpu().numpy().tolist()=}")
        soi_positions, eoi_positions, soa_positions, eoa_positions = locate_modal_range(sample_labels)
        if len(soi_positions) >= 1 and len(eoi_positions) >= 1:
            pred_img_start, pred_img_end = soi_positions[0], eoi_positions[0]
            if pred_img_start < pred_img_end:
                vision_loss = F.cross_entropy(
                    sample_logits[pred_img_start:pred_img_end + 1],
                    sample_labels[pred_img_start:pred_img_end + 1],
                    ignore_index=ignore_id
                )
                # print(f"{(sample_labels[pred_img_start:pred_img_end + 1].to(torch.long)).detach().cpu().numpy().tolist()=}")
                vision_logits.append(sample_logits[pred_img_start:pred_img_end + 1])
                vision_losses.append(vision_loss)
        if len(soa_positions) >= 1 and len(eoa_positions) >= 1:
            action_start, action_end = soa_positions[0], eoa_positions[0]
            if action_start < action_end:
                action_loss = F.cross_entropy(
                    sample_logits[action_start:action_end + 1],
                    sample_labels[action_start:action_end + 1],
                    ignore_index=ignore_id
                )
                # print(f"{(sample_labels[action_start:action_end + 1].to(torch.long)).detach().cpu().numpy().tolist()=}")
                action_logits.append(sample_logits[action_start:action_end + 1])
                action_losses.append(action_loss)
    total_vision_loss = torch.stack(vision_losses).mean() if vision_losses else torch.tensor(0.0, device=device)
    total_action_loss = torch.stack(action_losses).mean() if action_losses else torch.tensor(0.0, device=device)
    return (vision_logits, total_vision_loss), (action_logits, total_action_loss)


class MMaDAVLAModelLM(MMaDAModelLM):
    config_class = MMaDAVLAConfig
    base_model_prefix = "model"
    def __init__(self, config: MMaDAVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    def forward_process(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        logits = self(input_ids, attention_bias=attention_bias).logits
        # (vision_logits, vision_loss), (action_logits, action_loss) = calc_ti2ia_loss(logits, labels)
        # loss = action_loss + vision_loss
        loss = F.cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]),
            labels.contiguous().view(-1),
            ignore_index=ignore_id
        )
        # return (vision_logits, vision_loss), (action_logits, action_loss), loss
        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        steps: int = 64,
        block_len: int = 256,
        mask_token_id: int = 126336,
        prompt: Prompting = None,
        remasking: str = "low_confidence",
        **kwargs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1) if (attention_mask is not None and 0.0 in attention_mask) else None
        batch_size = input_ids.size(0)
        x = torch.full((batch_size, input_ids.shape[1] + max_new_tokens), mask_token_id, dtype=torch.long).to(input_ids.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        num_blocks = max_new_tokens // block_len
        steps = steps // num_blocks
        for num_block in range(num_blocks):
            block_mask_index = (x[:, input_ids.shape[1] + num_block * block_len:input_ids.shape[1] + (num_block + 1) * block_len:] == mask_token_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_token_id)
                logits = self(x, attention_bias=attention_bias).logits
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == "random":
                    x0_p = torch.rand_like(x0, device=x0.device)
                else:
                    raise NotImplementedError(remasking)
                x0_p[:, input_ids.shape[1] + (num_block + 1) * block_len:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
        x = x[:, input_ids.shape[1]:]
        images, actions = [], []
        for i in range(x.shape[0]):
            item_sequence = x[i]
            # print(f"{(len(prompt.tokenizer) + prompt.vision_codebook_size)=}")
            # print(f"{item_sequence.detach().cpu().numpy().tolist()=}")
            soi_positions, eoi_positions, soa_positions, eoa_positions = locate_modal_range(item_sequence)
            if len(soi_positions) >= 1 and len(eoi_positions) >= 1:
                pred_img_start, pred_img_end = soi_positions[0] + 1, eoi_positions[0]
                if pred_img_start < pred_img_end:
                    image = item_sequence[pred_img_start:pred_img_end] - len(prompt.tokenizer)
                    images.append(image)
            if len(soa_positions) >= 1 and len(eoa_positions) >= 1:
                action_start, action_end = soa_positions[0] + 1, eoa_positions[0]
                if action_start < action_end:
                    action = item_sequence[action_start:action_end] - (len(prompt.tokenizer) + prompt.vision_codebook_size)
                    actions.append(action)
        return images, actions