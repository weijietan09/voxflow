"""向量场估计器：给定 (x_t, t, 条件) 预测 mel 上的向量场 / 噪声。

结构是一个 1D U-Net。条件包括文本编码器给出的均值 μ（逐帧，与 mel 同长）
与说话人 embedding（沿时间广播），二者与 x_t 在通道维拼接后送入网络；
时间步 t 经正弦嵌入 + MLP 后注入每个残差块。

条件流匹配（CFM）与扩散共用这个估计器，区别只在训练目标与采样过程。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from voxflow.models.modules import ResnetBlock1D, SinusoidalPosEmb, TransformerBlock


@dataclass
class EstimatorConfig:
    """向量场估计器超参数。"""

    n_mels: int = 80
    hidden: int = 128
    spk_dim: int = 256
    time_dim: int = 128
    depth: int = 2
    heads: int = 4


class Downsample(nn.Module):
    """步长 2 的卷积下采样，时间维长度减半（向上取整）。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """最近邻插值到指定长度后再卷积，用于对齐 skip 连接的时间维。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, length: int) -> torch.Tensor:
        x = F.interpolate(x, size=length, mode="nearest")
        return self.conv(x)


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(channels, max_groups), channels)


class VectorFieldEstimator(nn.Module):
    """1D U-Net，预测与 mel 同形状的向量场 / 噪声。"""

    def __init__(self, config: EstimatorConfig | None = None) -> None:
        super().__init__()
        cfg = config or EstimatorConfig()
        self.config = cfg
        in_ch = 2 * cfg.n_mels + cfg.spk_dim  # x_t 与 μ 各 n_mels，再拼上说话人向量

        self.time_pos = SinusoidalPosEmb(cfg.time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.time_dim * 4),
            nn.Mish(),
            nn.Linear(cfg.time_dim * 4, cfg.time_dim),
        )

        self.in_conv = nn.Conv1d(in_ch, cfg.hidden, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList(
            [ResnetBlock1D(cfg.hidden, cfg.hidden, cfg.time_dim) for _ in range(cfg.depth)]
        )
        self.downsamples = nn.ModuleList([Downsample(cfg.hidden) for _ in range(cfg.depth)])

        self.mid_block1 = ResnetBlock1D(cfg.hidden, cfg.hidden, cfg.time_dim)
        self.mid_attn = TransformerBlock(cfg.hidden, heads=cfg.heads)
        self.mid_block2 = ResnetBlock1D(cfg.hidden, cfg.hidden, cfg.time_dim)

        self.upsamples = nn.ModuleList([Upsample(cfg.hidden) for _ in range(cfg.depth)])
        self.up_blocks = nn.ModuleList(
            [ResnetBlock1D(2 * cfg.hidden, cfg.hidden, cfg.time_dim) for _ in range(cfg.depth)]
        )

        self.out_norm = _group_norm(cfg.hidden)
        self.out_act = nn.Mish()
        self.out_conv = nn.Conv1d(cfg.hidden, cfg.n_mels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mu: torch.Tensor,
        spk: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数
        ----
        x : (B, n_mels, T) 当前带噪 / 插值的 mel
        t : (B,) 时间步，取值 [0, 1]
        mu : (B, n_mels, T) 文本条件（逐帧均值）
        spk : (B, spk_dim) 说话人 embedding
        """
        t_emb = self.time_mlp(self.time_pos(t))
        spk_broadcast = spk.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        h = torch.cat([x, mu, spk_broadcast], dim=1)
        h = self.in_conv(h)

        skips: list[torch.Tensor] = []
        lengths: list[int] = []
        for block, down in zip(self.down_blocks, self.downsamples, strict=True):
            h = block(h, t_emb)
            skips.append(h)
            lengths.append(h.shape[-1])
            h = down(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h.transpose(1, 2)).transpose(1, 2)
        h = self.mid_block2(h, t_emb)

        for up, block in zip(self.upsamples, self.up_blocks, strict=True):
            h = up(h, lengths.pop())
            h = torch.cat([h, skips.pop()], dim=1)
            h = block(h, t_emb)

        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)
