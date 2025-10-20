import torch
from typing import List, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer


ignore_id = -100
special_token_mappings = {
    "<|soi|>": 126084, "<|eoi|>": 126085, "<|sov|>": 126086, "<|eov|>": 126087, "<|t2i|>": 126088, "<|mmu|>": 126089, "<|t2v|>": 126090,
    "<|v2v|>": 126091, "<|lvg|>": 126092, "[iPAD]": 126093, "<|r2i|>": 126094, "<|soa|>": 126095, "<|eoa|>": 126096, "<|ti2ia|>": 126097
}


def truncate_trailing_padding(action_tokens: torch.Tensor, action_labels: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    non_padding_mask = action_tokens != ignore_id
    if not non_padding_mask.any():
        return action_tokens[:0], (action_labels[:0] if action_labels is not None else None)
    last_valid_idx = non_padding_mask.nonzero(as_tuple=True)[0][-1].item()
    truncated_tokens = action_tokens[:last_valid_idx + 1]
    truncated_labels = action_labels[:last_valid_idx + 1] if action_labels is not None else None
    return truncated_tokens, truncated_labels


class Prompting(object):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_text_len: int,
        mask_id: int,
        vision_codebook_size: int,
        action_codebook_size: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.vision_codebook_size = vision_codebook_size
        self.action_codebook_size = action_codebook_size

        self.sptids_dict = {}
        for token, token_id in special_token_mappings.items():
            self.sptids_dict[token] = torch.tensor([token_id])
        self.sptids_dict["<|sot|>"] = torch.tensor([self.tokenizer.bos_token_id])
        self.sptids_dict["<|eot|>"] = torch.tensor([self.tokenizer.eos_token_id])
        end_header_tokens = self.tokenizer.convert_tokens_to_ids(["<|end_header_id|>"])
        self.sptids_dict["<|end_header_id|>"] = torch.tensor(end_header_tokens)
        self.sptids_dict["<|eot_id|>"] = torch.tensor(self.tokenizer.convert_tokens_to_ids(["<|eot_id|>"]))
        self.sptids_dict["<|start_header_id|>"] = torch.tensor(self.tokenizer.convert_tokens_to_ids(["<|start_header_id|>"]))
        self.max_text_len = max_text_len
        self.mask_id = mask_id
        self.pad_id = special_token_mappings["[iPAD]"]
        self.ignore_id = ignore_id

    def training(
        self,
        task_inst: List[str],
        cur_image_tokens: torch.Tensor,
        pred_image_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        pred_image_labels: torch.Tensor = None,
        action_labels: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert cur_image_tokens.device == pred_image_tokens.device == action_tokens.device
        device = cur_image_tokens.device
        text_ids = self.tokenizer(task_inst).input_ids
        input_ids, attn_mask, labels = [], [], []
        for i in range(len(text_ids)):
            text_tokens = ([self.tokenizer.bos_token_id] if len(text_ids[i]) == 0 else [self.tokenizer.bos_token_id] + text_ids[i]) + [self.tokenizer.eos_token_id]
            if self.max_text_len >= len(text_tokens):
                text_mask = [0] * (self.max_text_len - len(text_tokens)) + [1] * len(text_tokens)
                text_tokens = [self.pad_id] * (self.max_text_len - len(text_tokens)) + text_tokens
            else:
                text_tokens = text_tokens[:self.max_text_len - 1] + [self.tokenizer.eos_token_id]
                text_mask = [1] * len(text_tokens)
            # prompting -- [task token] [soi] [cur image tokens] [eoi] [sot] [text tokens] [eot] [soi] [pred image tokens] [eoi] [soa] [action tokens] [eoa]
            action_tokens_item, action_labels_item = action_tokens[i], (action_labels[i] if action_labels is not None else None)
            action_tokens_item, action_labels_item = truncate_trailing_padding(action_tokens_item, action_labels_item)
            item_labels = torch.cat([
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(cur_image_tokens[i]) * self.ignore_id,
                torch.tensor([self.ignore_id]).to(device),
                torch.ones(len(text_tokens)).to(dtype=torch.long, device=device) * self.ignore_id,
                self.sptids_dict["<|soi|>"].to(device),
                pred_image_labels[i],
                self.sptids_dict["<|eoi|>"].to(device),
                self.sptids_dict["<|soa|>"].to(device),
                action_labels_item,
                self.sptids_dict["<|eoa|>"].to(device),
            ], dim=0)
            item_input_ids = torch.cat([
                self.sptids_dict["<|ti2ia|>"].to(device),
                self.sptids_dict["<|soi|>"].to(device),
                cur_image_tokens[i],
                self.sptids_dict["<|eoi|>"].to(device),
                torch.tensor(text_tokens).to(device),
                torch.tensor([self.mask_id]).to(device),
                pred_image_tokens[i],
                torch.tensor([self.mask_id]).to(device),
                torch.tensor([self.mask_id]).to(device),
                action_tokens_item,
                torch.tensor([self.mask_id]).to(device),
            ], dim=0)
            item_attn_mask = [1] + [1] + [1] * cur_image_tokens.shape[-1] + [1] + text_mask
            item_attn_mask += [1] + [1] * pred_image_tokens.shape[-1] + [1] + [1] + [1] * action_tokens_item.shape[-1] + [1]
            item_labels = torch.where(item_labels == self.pad_id, self.ignore_id, item_labels)
            labels.append(item_labels.unsqueeze(-1))
            input_ids.append(item_input_ids.unsqueeze(-1))
            attn_mask.append((torch.tensor(item_attn_mask).to(device)).unsqueeze(-1))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id).squeeze(-1)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0).squeeze(-1)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.ignore_id).squeeze(-1)
        return input_ids, attn_mask, labels
    
    
    def inference(self, task_inst: List[str], cur_image_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device, text_ids = cur_image_tokens.device, self.tokenizer(task_inst).input_ids
        input_ids, attn_mask = [], []
        for i in range(len(text_ids)):
            text_tokens = ([self.tokenizer.bos_token_id] if len(text_ids[i]) == 0 else [self.tokenizer.bos_token_id] + text_ids[i]) + [self.tokenizer.eos_token_id]
            if self.max_text_len >= len(text_tokens):
                text_mask = [0] * (self.max_text_len - len(text_tokens)) + [1] * len(text_tokens)
                text_tokens = [self.pad_id] * (self.max_text_len - len(text_tokens)) + text_tokens
            else:
                text_tokens = text_tokens[:self.max_text_len - 1] + [self.tokenizer.eos_token_id]
                text_mask = [1] * len(text_tokens)
            item_input_ids = torch.cat([
                self.sptids_dict["<|ti2ia|>"].to(device),
                self.sptids_dict["<|soi|>"].to(device),
                cur_image_tokens[i],
                self.sptids_dict["<|eoi|>"].to(device),
                torch.tensor(text_tokens).to(device),
            ], dim=0)
            item_attn_mask = [1] + [1] + [1] * cur_image_tokens.shape[-1] + [1] + text_mask
            input_ids.append(item_input_ids.unsqueeze(-1))
            attn_mask.append((torch.tensor(item_attn_mask).to(device)).unsqueeze(-1))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id).squeeze(-1)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0).squeeze(-1)
        return input_ids, attn_mask