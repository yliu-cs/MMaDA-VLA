import torch
import autoroot
from typing import Tuple
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
    
    @torch.no_grad()
    def prepare_inputs_and_labels(
        pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
        texts: Union[str, str],
        is_train: bool = True,
    ):
        pass
    
    def forward_process(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        max_seq_len: int = 128,
        p_mask: torch.Tensor = None,
        answer_length: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        logits = self(input_ids, attention_bias=attention_bias).logits