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
