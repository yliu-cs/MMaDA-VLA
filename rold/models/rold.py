import torch
import autoroot
from transformers import PretrainedConfig
from rold.models.llada import LLaDAModelLM


class RoLDConfig(PretrainedConfig):
    model_type = "rold"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            "codebook_size",
            "num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size"
        ]
        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])


class RoLDModelLM(LLaDAModelLM):
    config_class = RoLDConfig
    base_model_prefix = "model"
    def __init__(self, config: RoLDConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)