import os
import math
import torch
import warnings
from enum import StrEnum
from torch import nn, einsum
from functools import partial
from abc import abstractmethod
import torch.nn.functional as F
from dataclasses import dataclass, fields
from transformers.cache_utils import Cache
from collections.abc import MutableMapping
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Tuple, Dict, Sequence, Any, Union, Optional, Callable, Iterable, NamedTuple, cast


class LayerNormType(StrEnum):
    default = "default"
    low_precision = "low_precision"
    rms = "rms"
    gemma_rms = "gemma_rms"
    amd_compatible = "amd_compatible"


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"
    parallel = "parallel"
    llama = "llama"


class InitFnType(StrEnum):
    mitchell = "mitchell"
    normal = "normal"
    kaiming_normal = "kaiming_normal"
    fan_in = "fan_in"
    full_megatron = "full_megatron"


class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    two_in_three = "two_in_three"
    three_in_four = "three_in_four"
    four_in_five = "four_in_five"
    nine_in_ten = "nine_in_ten"
    fine_grained = "fine_grained"


@dataclass
class ModelConfig():
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    n_layers: int = 12
    mlp_ratio: int = 4
    mlp_hidden_size: Optional[int] = None
    activation_type: ActivationType = ActivationType.swiglu
    block_type: BlockType = BlockType.sequential
    block_group_size: int = 1
    alibi: bool = False
    alibi_bias_max: float = 8.0
    rope: bool = False
    rope_full_precision: bool = True
    flash_attention: bool = False
    attention_dropout: float = 0.1
    multi_query_attention: Optional[bool] = None
    attention_layer_norm: bool = False
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    input_emb_norm: bool = False
    layer_norm_type: LayerNormType = LayerNormType.default
    layer_norm_with_affine: bool = True
    rms_norm_eps: float = 1e-05
    attention_layer_norm_with_affine: bool = True
    max_sequence_length: int = 1024
    rope_theta: float = 10000.0
    include_qkv_bias: Optional[bool] = False
    include_bias: bool = False
    bias_for_layer_norm: Optional[bool] = None
    scale_logits: bool = False
    vocab_size: int = 50257
    embedding_size: Optional[int] = 50304
    weight_tying: bool = True
    eos_token_id: int = 50256
    pad_token_id: int = 50256
    mask_token_id: Optional[int] = 50256
    init_device: Optional[str] = None
    init_fn: InitFnType = InitFnType.normal
    init_std: float = 0.02
    init_cutoff_factor: Optional[float] = None
    precision: Optional[str] = None

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            return 1 if self.multi_query_attention is True else self.n_heads
        else:
            if self.multi_query_attention is None:
                return self.n_kv_heads
            n_kv_heads_should_be = 1 if self.multi_query_attention else self.n_heads
            if self.n_kv_heads == n_kv_heads_should_be:
                return n_kv_heads_should_be
            else:
                raise Exception("You can't set `multi_query_attention` and `n_kv_heads` at the same time.")


class LLaDAConfig(PretrainedConfig):
    model_type = "llada"
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(self, use_cache: bool = False, **kwargs):
        model_config = ModelConfig()
        all_kwargs = model_config.__dict__
        all_kwargs.update(kwargs)
        all_kwargs.update({"use_cache": use_cache})
        all_kwargs.update({"architectures": all_kwargs.get("architectures", ["LLaDAModelLM"])})
        super().__init__(**all_kwargs)
    
    @property
    def num_attention_heads(self) -> int:
        return self.n_heads
    
    @property
    def num_hidden_layers(self) -> int:
        return self.n_layers
    
    @property
    def hidden_size(self) -> int:
        return self.d_model


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: Optional[float] = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.kaiming_normal:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the '{InitFnType.full_megatron}' init, every module must have a type.")
        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3
        if type_of_module == ModuleType.in_module:
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            std = config.d_model ** -0.5
        else:
            raise RuntimeError(f"Unknown module type: '{type_of_module}'")
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_factor * std, b=cutoff_factor * std)
    else:
        raise NotImplementedError(f"Unknown init function: '{config.init_fn}'")
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False) -> None:
    if check_neg_inf:
        x.mask_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.mask_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def activation_checkpoint_function(config: ModelConfig) -> Callable:
    preserve_rng_state = (config.attention_dropout == 0.0) and (config.embedding_dropout == 0.0) and (config.residual_dropout == 0.0)
    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    pass


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input if self.p == 0.0 else F.dropout(input, p=self.p, training=self.training, inplace=self.inplace)


class LayerNormBase(nn.Module):
    def __init__(   
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ) -> None:
        super().__init__()
        self.config, self.eps, self.normalized_shape = config, eps, (size or config.d_model, )
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=self.config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=self.config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> "LayerNormBase":
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")
    
    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if tensor.device.type == "cuda" and torch.cuda.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.cuda.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class LayerNorm(LayerNormBase):
    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ) -> None:
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps)
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ) -> None:
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)
        if self.weight is not None:
            if self.bias is not None:
                x = x * self.weight + self.bias
            else:
                x = x * self.weight
        return x


class GemmaRMSLayerNorm(LayerNormBase):
    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ) -> None:
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)
        if self.weight is not None:
            if self.bias is not None:
                x = x * (self.weight + 1) + self.bias
            else:
                x = x * (self.weight + 1)
        return x


class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, cache: BufferCache) -> None:
        super().__init__()
        self.config, self.__cache = config, cache
        self.rope_theta = config.rope_theta
        self.get_rotary_embedding(self.config.max_sequence_length, _non_meta_init_device(self.config))
    
    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]
        
        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i, j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_, k_ = (q.float(), k.float()) if self.config.rope_full_precision else (q, k)
        with torch.autocast(device_type=q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(query_len, q.device)
            pos_sin, pos_cos = pos_sin.type_as(q_), pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(pos_sin[:, :, key_len - query_len:key_len, :], pos_cos[:, :, key_len - query_len:key_len, :], q_)
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class Activation(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError
    
    @classmethod
    def build(cls, config: ModelConfig) -> "Activation":
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation type: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
    
    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_mask(seq_len: int, device: torch.device) -> torch.FloatTensor:
    attn_bias = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.float), diagonal=1)
    attn_bias.masked_fill_(attn_bias == 1, torch.finfo(attn_bias.dtype).min)
    return attn_bias.view(1, 1, seq_len, seq_len)


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_mask(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads)))


class LLaDABlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: ModelConfig,
        cache: BufferCache,
    ) -> None:
        super().__init__()
        self.layer_id, self.config, self.__cache = layer_id, config, cache
        self.hidden_size = (self.config.mlp_hidden_size if self.config.mlp_hidden_size is not None else self.config.d_model * self.config.mlp_ratio)
        assert self.config.d_model % self.config.n_heads == 0, "d_model must be divisible by n_heads"
        self._activation_checkpoint_fn = None
        self.dropout = Dropout(self.config.residual_dropout)
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if self.config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(self.config.d_model // self.config.n_heads) * self.config.effective_n_kv_heads,
                elementwise_affine=self.config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=self.config.attention_layer_norm_with_affine)
        self.act = Activation.build(self.config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0, "hidden_size must be divisible by act.output_multiplier"
        self.attn_out = nn.Linear(self.config.d_model, self.config.d_model, bias=self.config.include_bias, device=self.config.init_device)
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            self.config.d_model,
            bias=self.config.include_bias,
            device=self.config.init_device,
        )
        self.ff_out._is_residual = True
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(self.config, self.__cache)
        self.flash_attn_func = None
        if self.config.flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass
    
    def reset_parameters(self) -> None:
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(self.config, self.attn_out, d=self.config.d_model, layer_id=self.layer_id, type_of_module=ModuleType.out_module)
        init_weights(self.config, self.ff_out, d=self.ff_out.in_features, layer_id=self.layer_id, type_of_module=ModuleType.out_module)
    
    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy] = None) -> None:
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config) if strategy == ActivationCheckpointingStrategy.fine_grained else None
    
    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        if bias.device.type == "cuda" and torch.cuda.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.cuda.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias
    
    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=False)
            return r.transpose(1, 2)
        else:
            assert k.size(1) == v.size(1), "k and v must have the same number of heads"
            num_kv_heads, num_q_heads = k.size(1), q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False)
    
    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()
        dtype = k.dtype
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q).to(dtype=dtype), self.k_norm(k).to(dtype=dtype)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        if layer_past is not None:
            past_key, past_value = layer_past
            k, v = torch.cat((past_key, k), dim=-2), torch.cat((past_value, v), dim=-2)
        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]
        if self.config.rope:
            q, k = self.rotary_emb(q, k)
        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(attention_bias[:, :, key_len - query_len:key_len, :key_len], dtype)
        attn = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_out(attn), present
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError
    
    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> "LLaDABlock":
        if config.block_type == BlockType.sequential:
            return LLaDASequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class LLaDASequentialBlock(LLaDABlock):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache) -> None:
        super().__init__(layer_id, config, cache)
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim
        )
        self.attn_proj = nn.Linear(
            config.d_model,
            sum(self.fused_dims),
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )
    
    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(self.config, self.attn_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self._activation_checkpoint_fn is not None:
            q, k, v = self.attn_proj(self._activation_checkpoint_fn(self.attn_norm, x)).split(self.fused_dims, dim=-1)
        else:
            q, k, v = self.attn_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)
        if self._activation_checkpoint_fn is not None:
            attn, cache = self._activation_checkpoint_fn(self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)
        else:
            attn, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)
        x = x + self.dropout(attn)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)
        x = og_x + self.dropout(self.ff_out(x))
        return x, cache


class LLaDALlamaBlock(LLaDABlock):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache) -> None:
        super().__init__(layer_id, config, cache)
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device)
        self.k_proj = nn.Linear(config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device)
        self.v_proj = nn.Linear(config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device)
        self.ff_proj = nn.Linear(config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device)
        self.up_proj = nn.Linear(config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device)
    
    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        x_normed = self.attn_norm(x)
        q, k, v = self.q_proj(x_normed), self.k_proj(x_normed), self.v_proj(x_normed)
        if self._activation_checkpoint_fn is not None:
            attn, cache = self._activation_checkpoint_fn(self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)
        else:
            attn, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)
        x = x + self.dropout(attn)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)
        else:
            x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)
        x = x * x_up
        x = og_x + self.dropout(self.ff_out(x))
        return x, cache


class LLaDAOutput(NamedTuple):
    logits: torch.FloatTensor
    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    hidden_states: Optional[Tuple[torch.Tensor]]


class LLaDAGenerationOutput(NamedTuple):
    token_ids: torch.LongTensor
    scores: torch.FloatTensor


class LLaDABlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None) -> None:
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layer_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if (
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer) or
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0) or
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0) or
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
            ):
                x, cache = self._activation_checkpoint_fn(block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
            else:
                x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
            if attn_key_values is not None:
                assert cache is not None, "cache should not be None if use_cache is True"
                attn_key_values.append(cache)
        return x, attn_key_values
    
    def reset_parameters(self) -> None:
        for block in self:
            block.reset_parameters()
    
    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy] = None) -> None:
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class LLaDAModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = False) -> None:
        super().__init__()
        self.config = config
        self.__cache = BufferCache()
        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")
        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")
        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding_size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 == 0:
                warnings.warn("Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning)
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)
        if not (0 < self.config.block_group_size <= self.config.n_layers and self.config.n_layers % self.config.block_group_size == 0):
            raise Exception("n layers must be divisible by block group size")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(self.config.embedding_size or self.config.vocab_size, self.config.d_model, device=self.config.init_device),
            "emb_drop": Dropout(self.config.embedding_dropout),
            "ln_f": LayerNorm.build(self.config)
        })
        blocks = [LLaDABlock.build(i, self.config, self.__cache) for i in range(self.config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                LLaDABlockGroup(self.config, i, blocks[i:i + self.config.block_group_size])
                for i in range(0, self.config.n_layers, self.config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})
        if not (self.config.alibi or self.config.rope):
            self.transformer.update({"wpe": nn.Embedding(self.config.max_sequence_length, self.config.d_model, device=self.config.init_device)})
        if not self.config.weight_tying:
            self.transformer.update({
                "ff_out": nn.Linear(
                    self.config.d_model,
                    self.config.embedding_size or self.config.vocab_size,
                    bias=self.config.include_bias,
                    device=self.config.init_device
                )
            })
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None
        if self.config.alibi:
            get_causal_attention_bias(self.__cache, self.config.max_sequence_length, _non_meta_init_device(self.config))
            self.get_alibi_attention_bias(self.config.max_sequence_length, _non_meta_init_device(self.config))
    
    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy] = None) -> None:
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)
    
    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else: 
            return device
    
    def reset_parameters(self) -> None:
        init_weights(self.config, self.transformer.wte, std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0)
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)
        self.transformer.ln_f.reset_parameters()
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()
    
    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[-1] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
    ) -> LLaDAOutput:
        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        assert (past_key_values is None and not use_cache), "The kvcache is not suppotred for MDM."
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        if past_key_values:
            assert len(past_key_values) == self.config.n_layers, "past_key_values must be a list of length n_layers"
        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings
        if self.config.input_emb_norm:
            x = x * (self.config.d_model ** 0.5)
        if not (self.config.alibi or self.config.rope):
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = pos_emb + x
        x = self.transformer.emb_drop(x)
        if attention_mask is not None and 0.0 in attention_mask:
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None
        if attention_bias is not None or attention_mask is not None or self.config.alibi or past_key_values is not None:
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        all_hidden_states = []
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    all_hidden_states.append(x)
                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if (
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer) or
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0) or
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0) or
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
                ):
                    x, cache = self._activation_checkpoint_fn(block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
                else:
                    x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
                if attn_key_values is not None:
                    assert cache is not None, "cache should not be None if use_cache is True"
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    all_hidden_states.append(x)
                layer_past = None if past_key_values is None else past_key_values[group_idx * self.config.block_group_size:(group_idx + 1) * self.config.block_group_size]
                x, cache = block_group(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
                if attn_key_values is not None:
                    assert cache is not None, "cache should not be None if use_cache is True"
                    attn_key_values.append(cache)
        if last_logits_only:
            x = x[:, -1, :].unsqueeze(1)
        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(x)
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))
        return LLaDAOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None
        )


class LLaDAModelLM(PreTrainedModel):
    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["LlaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock"]
    def __init__(
        self,
        config: LLaDAConfig,
        model: Optional[LLaDAModel] = None,
        init_params: bool = False
    ) -> None:
        super().__init__(config)
        if model:
            self.model = model
        else:
            kwargs = {}
            for field in fields(ModelConfig):
                kwargs[field.name] = getattr(config, field.name)
            model_config = ModelConfig(**kwargs)
            model_config.init_device = "cpu"
            self.model = LLaDAModel(model_config, init_params=init_params)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Cache] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache
        if output_attentions:
            raise ValueError("output_attentions is not supported for LLaDA.")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=None,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
        logits, hidden_states, loss = outputs.logits, outputs.hidden_states, None
        if labels is not None:
            warnings.warn("Note that for LLaDA, you cannot calculate the loss here.", UserWarning)
        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )
    
    def can_generate(self) -> bool:
        return True
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if past_key_values:
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}
        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs
    
    def get_input_embeddings(self) -> nn.Module:
        return self.model.transformer.wte
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.transformer.wte = value
    
    def get_output_embeddings(self) -> nn.Module:
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out
    
    def set_output_embeddings(self, value: nn.Module) -> None:
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value
    
    def tie_weights(self) -> None:
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte


if __name__ == "__main__":
    llada_path = os.path.join(os.sep, "ssdwork", "liuyang", "Models", "LLaDA-8B-Instruct")
    model = LLaDAModelLM.from_pretrained(llada_path)
    print(model)