import os
import math
import torch
import numpy as np
from torch import nn
from random import sample
from einops import rearrange
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualFSQ
from transformers import PretrainedConfig, PreTrainedModel


class ActionRVQConfig(PretrainedConfig):
    model_type = "ActionRVQModel"
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        allowed_keys = [
            "dims",
            "levels",
            "num_quantizers"
        ]
        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])


class ResLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.layer_norm(self.linear2(x) + x)
        return x


class ActionRVQModel(PreTrainedModel):
    config_class = ActionRVQConfig
    def __init__(self, config: ActionRVQConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.encoder, self.decoder = nn.Sequential(), nn.Sequential()
        for i in range(len(self.config.dims) - 1):
            if i == 0 or i == len(self.config.dims) - 2:
                self.encoder.append(nn.Linear(self.config.dims[i], self.config.dims[i + 1]))
                self.decoder.append(nn.Linear(self.config.dims[i + 1], self.config.dims[i]))
            else:
                self.encoder.append(nn.Sequential(nn.ReLU(), ResLinear(self.config.dims[i], self.config.dims[i + 1])))
                self.decoder.append(nn.Sequential(ResLinear(self.config.dims[i + 1], self.config.dims[i]), nn.ReLU()))
        self.decoder = nn.Sequential(*reversed([layer for layer in self.decoder]))
        self.vq = ResidualFSQ(dim=self.config.dims[-1], levels=self.config.levels, num_quantizers=self.config.num_quantizers)
        setattr(self.config, "codebook_size", self.vq.codebook_size)
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        quantized, _ = self.vq(self.encoder(action))
        action_recon = self.decoder(quantized)
        loss = F.mse_loss(action, action_recon)
        return loss, action_recon
    
    @torch.no_grad()
    def tokenize(self, action: torch.Tensor) -> torch.Tensor:
        _, indices = self.vq(self.encoder(action))
        return indices.view(indices.shape[0], -1)
    
    @torch.no_grad()
    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.view(indices.shape[0], int(indices.shape[1] ** 0.5), int(indices.shape[1] ** 0.5))
        return self.decoder(self.vq.get_output_from_indices(indices))


if __name__ == "__main__":
    data_path = os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN", "calvin_abc_d_action_8step.npy")
    actions = np.load(data_path)
    act = torch.from_numpy(actions[sample(list(range(actions.shape[0])), 4)]).float()
    act_rvq_model = ActionRVQModel(ActionRVQConfig(dims=[7, 2048, 2048, 2048, 512], levels=[8, 5, 5, 5], num_quantizers=8))
    print(f"{act_rvq_model(act)[0].item()=}")
    print(f"{act.shape=}")
    act_ids = act_rvq_model.tokenize(act)
    recon_act = act_rvq_model.detokenize(act_ids)
    print(f"{recon_act.shape=}")
    mse_loss = F.mse_loss(act, recon_act)
    print(f"{mse_loss.item()=}")