import os
import math
import torch
import autoroot
import numpy as np
from torch import nn
from PIL import Image
from typing import List, Tuple
import torch.nn.functional as F
from dataclasses import dataclass, field
from mmadavla.models.utils import ModelMixin
from mmadavla.data.utils import image_transform
from torch.distributions.categorical import Categorical
from diffusers.configuration_utils import ConfigMixin, register_to_config


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class AttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
            eps=1e-6,
            affine=True
        )
        self.q = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        h = x
        h = self.norm(h)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W)
        q = q.permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, -1)
        w = F.softmax(torch.bmm(q, k) * (int(C) ** -0.5), dim=2)
        v = v.reshape(B, C, H * W)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w)
        h = h.reshape(B, C, H, W)
        h = self.proj_out(h)
        return x + h


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = nn.GroupNorm(
            num_groups=32,
            num_channels=self.in_channels,
            eps=1e-6,
            affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.temb_proj = nn.Linear(
            in_features=temb_channels,
            out_features=self.out_channels
        ) if temb_channels > 0 else None
        self.norm2 = nn.GroupNorm(
            num_groups=32,
            num_channels=self.out_channels,
            eps=1e-6,
            affine=True
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
    
    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor
    ) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h


class VQGANEncoder(ModelMixin, ConfigMixin):
    @dataclass
    class Config:
        ch: int = 128
        ch_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 4, 4])
        num_res_blocks: List[int] = field(default_factory=lambda: [4, 3, 4, 3, 4])
        attn_resolutions: List[int] = field(default_factory=lambda: [5])
        dropout: float = 0.0
        in_ch: int = 3
        out_ch: int = 3
        resolution: int = 256
        z_channels: int = 13
        double_z: bool = False
    
    def __init__(
        self,
        ch: int = 128,
        ch_mult: List[int] = [1, 2, 2, 4, 4],
        num_res_blocks: List[int] = [4, 3, 4, 3, 4],
        attn_resolutions: List[int] = [5],
        dropout: float = 0.0,
        in_ch: int = 3,
        out_ch: int = 3,
        resolution: int = 256,
        z_channels: int = 13,
        double_z: bool = False
    ) -> None:
        super().__init__()
        self.in_ch, self.ch, self.temb_ch, self.num_resolutions = in_ch, ch, 0, len(ch_mult)
        self.num_res_blocks, self.resolution = num_res_blocks, resolution
        self.conv_in = torch.nn.Conv2d(self.in_ch, self.ch, kernel_size=3, stride=1, padding=1)
        cur_res, in_ch_mult = self.resolution, (1, ) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block, attn = nn.ModuleList(), nn.ModuleList()
            block_in, block_out = self.ch * in_ch_mult[i_level], self.ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dropout=dropout,
                    temb_channels=self.temb_ch,
                ))
                block_in = block_out
                if cur_res in attn_resolutions:
                    attn.append(AttnBlock(in_channels=block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(in_channels=block_in, with_conv=True)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(in_channels=block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.norm_out = nn.GroupNorm(
            num_groups=32,
            num_channels=block_in,
            eps=1e-6,
            affine=True
        )
        self.conv_out = nn.Conv2d(
            in_channels=block_in,
            out_channels=2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.quant_conv = nn.Conv2d(
            in_channels=z_channels,
            out_channels=z_channels,
            kernel_size=1,
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        temb = None
        hs = [self.conv_in(x)]
        for i_level, down in enumerate(self.down):
            for i_block in range(self.num_res_blocks[i_level]):
                h = down.block[i_block](hs[-1], temb)
                if len(down.attn) > 0:
                    h = down.attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(down.downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class LFQuantizer(nn.Module):
    def __init__(
        self,
        num_codebook_entry: int = -1,
        codebook_dim: int = 13,
        beta: float = 0.25,
        entropy_multiplier: float = 0.1,
        commit_loss_multiplier: float = 0.1
    ) -> None:
        super().__init__()
        self.codebook_size, self.e_dim, self.beta = 2 ** codebook_dim, codebook_dim, beta
        indices = torch.arange(self.codebook_size)
        binary = (indices.unsqueeze(1) >> torch.arange(codebook_dim - 1, -1, -1, dtype=torch.long)) & 1
        embedding = binary.float() * 2 - 1
        self.register_buffer("embedding", embedding)
        self.register_buffer("power_vals", 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.commit_loss_multiplier, self.entropy_multiplier = commit_loss_multiplier, entropy_multiplier
    
    def get_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        return (self.power_vals.reshape(1, -1, 1, 1) * (z_q > 0).float()).sum(dim=1, keepdim=True).long()
    
    def get_codebook_entry(self, indices: torch.Tensor, shape: Tuple[int, int] = None) -> torch.Tensor:
        if shape is None:
            H, W = int(math.sqrt(indices.shape[-1])), int(math.sqrt(indices.shape[-1]))
        else:
            H, W = shape
        B, _ = indices.shape
        indices = indices.reshape(B, -1)
        z_q = self.embedding[indices].view(B, H, W, -1)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q
    
    def forward(self, z: torch.Tensor, get_code: bool = False) -> torch.Tensor:
        if get_code:
            return self.get_codebook_entry(z)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        ge_zero = (z_flattened > 0).float()
        ones = torch.ones_like(z_flattened)
        z_q = ones * ge_zero + -ones * (1 - ge_zero)
        z_q = z_flattened + (z_q - z_flattened).detach()
        logit = torch.stack([
                -(z_flattened - torch.ones_like(z_q)).pow(2),
                -(z_flattened - torch.ones_like(z_q) * -1).pow(2),
            ],
            dim=-1
        )
        cat_dist = Categorical(logits=logit)
        entropy = cat_dist.entropy().mean()
        mean_prob = cat_dist.probs.mean(0)
        mean_entropy = Categorical(probs=mean_prob).entropy().mean()
        commit_loss = torch.mean((z_q.detach() - z_flattened) ** 2) + self.beta * torch.mean((z_q - z_flattened.detach()) ** 2)
        z_q = z_q.view(z.shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return {
            "z": z_q,
            "quantizer_loss": commit_loss * self.commit_loss_multiplier,
            "entropy_loss": (entropy - mean_entropy) * self.entropy_multiplier,
            "indices": self.get_indices(z_q)
        }


class VQGANDecoder(ModelMixin, ConfigMixin):
    def __init__(
        self,
        ch: int = 128,
        ch_mult: List[int] = [1, 1, 2, 2, 4],
        num_res_blocks: List[int] = [4, 4, 3, 4, 3],
        attn_resolutions: List[int] = [5],
        dropout: float = 0.0,
        in_ch: int = 3,
        out_ch: int = 3,
        resolution: int = 256,
        z_channels: int = 13,
        double_z: bool = False
    ) -> None:
        super().__init__()
        self.in_ch, self.ch, self.temb_ch, self.num_resolutions = in_ch, ch, 0, len(ch_mult)
        self.num_res_blocks, self.resolution, self.give_pre_end = num_res_blocks, resolution, False
        self.z_channels = z_channels
        in_ch_mult = (1, ) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        cur_res = self.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, cur_res, cur_res)
        self.conv_in = nn.Conv2d(
            in_channels=z_channels,
            out_channels=block_in,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(in_channels=block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block, attn = nn.ModuleList(), nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dropout=dropout,
                    temb_channels=self.temb_ch,
                ))
                block_in = block_out
                if cur_res in attn_resolutions:
                    attn.append(AttnBlock(in_channels=block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(in_channels=block_out, with_conv=True)
            self.up.insert(0, up)
        self.norm_out = nn.GroupNorm(
            num_groups=32,
            num_channels=block_in,
            eps=1e-6,
            affine=True
        )
        self.conv_out = nn.Conv2d(
            in_channels=block_in,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.post_quant_conv = nn.Conv2d(
            in_channels=z_channels,
            out_channels=z_channels,
            kernel_size=1,
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.last_z_shape = z.shape
        temb = None
        z = self.post_quant_conv(z)
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class MagViTv2(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VQGANEncoder()
        self.decoder = VQGANDecoder()
        self.quantize = LFQuantizer()

    def forward(
        self
        , pixel_values: torch.Tensor
        , return_loss: bool = False
    ) -> None:
        pass

    def encode(
        self
        , pixel_values: torch.Tensor
        , return_loss: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_states = self.encoder(pixel_values)
        quantized_states = self.quantize(hidden_states)["z"]
        codebook_indices = self.quantize.get_indices(quantized_states).reshape(pixel_values.shape[0], -1)
        return (quantized_states, codebook_indices)

    def get_code(
        self,
        pixel_values: torch.Tensor
    ) -> List[torch.Tensor]:
        hidden_states = self.encoder(pixel_values)
        quantized_states = self.quantize(hidden_states)["z"]
        codebook_indices = self.quantize.get_indices(quantized_states).reshape(pixel_values.shape[0], -1)
        return codebook_indices

    def decode_code(
        self
        , codebook_indices: List[torch.Tensor]
        , shape: Tuple[int, int] = None
    ) -> torch.Tensor:
        z_q = self.quantize.get_codebook_entry(codebook_indices, shape=shape)
        reconstructed_pixel_values = self.decoder(z_q)
        return reconstructed_pixel_values


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq_path = os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2")
    model = MagViTv2.from_pretrained(vq_path).to(device)
    image_path = os.path.join(os.getcwd(), "media", "calvin.jpg")
    image = Image.open(image_path).convert("RGB")
    print(f"{np.array(image).shape=}")
    image = image_transform(image).unsqueeze(0).to(device)
    print(f"{image.shape=}")
    image = model.decode_code(model.get_code(image))
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0) * 255
    image = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image)
    image.save(os.path.join(os.getcwd(), "media", "recon_calvin.jpg"))