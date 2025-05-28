import torch
import autoroot
import torch.nn.functional as F
from typing import Tuple, Union
from rold.utils.prompt import ignore_id
from transformers import PretrainedConfig
from rold.models.mmada import MMaDAModelLM


class RoLDConfig(PretrainedConfig):
    model_type = "rold"
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
            "new_vocab_size"
        ]
        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])


class RoLDModelLM(MMaDAModelLM):
    config_class = RoLDConfig
    base_model_prefix = "model"
    def __init__(self, config: RoLDConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    def forward_process(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        logits = self(input_ids, attention_bias=attention_bias).logits
        loss = F.cross_entropy(logits.contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1), ignore_index=ignore_id)
        return logits, loss