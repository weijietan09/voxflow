"""声学模型通用组件。

包含时间步嵌入、带时间调制的 1D 残差块，以及一个极简的
Transformer 块（多头自注意力 + 前馈）。这些块被向量场估计器、
文本编码器等复用。
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """由长度向量生成布尔掩码，形状 ``(B, max_len)``，True 表示有效帧。"""
    max_len = int(max_len if max_len is not None else int(lengths.max()))
    ids = torch.arange(max_len, device=lengths.device)
    return ids.unsqueeze(0) < lengths.unsqueeze(1)


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """挑一个能整除 ``channels`` 的分组数，避免 GroupNorm 报错。"""
    return math.gcd(channels, max_groups)


class SinusoidalPosEmb(nn.Module):
    """正弦时间步 / 位置嵌入。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalPosEmb 的维度必须为偶数")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConvBlock1D(nn.Module):
    """Conv1d + GroupNorm + Mish 的基本卷积块。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(_num_groups(out_ch, groups), out_ch)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.act(self.norm(self.conv(x)))
        if mask is not None:
            x = x * mask
        return x


class ResnetBlock1D(nn.Module):
    """带时间步调制的 1D 残差块（扩散 / 流匹配的主力构件）。"""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.block1 = ConvBlock1D(in_ch, out_ch, groups=groups)
        self.block2 = ConvBlock1D(out_ch, out_ch, groups=groups)
        self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(time_dim, out_ch))
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.block1(x, mask)
        h = h + self.time_mlp(t).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x)


class MultiHeadSelfAttention(nn.Module):
    """标准多头自注意力，支持 key padding mask。"""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim 必须能被 heads 整除")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, c)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer 块：自注意力 + 前馈。"""

    def __init__(self, dim: int, heads: int = 4, ff_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x
