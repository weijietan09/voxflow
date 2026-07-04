"""时长预测器与长度调节（length regulator）。

时长预测器从文本编码器的隐状态回归每个音素的对数时长；
长度调节把逐音素的条件按时长在时间轴上展开成逐帧条件。
"""

from __future__ import annotations

import torch
from torch import nn


class DurationPredictor(nn.Module):
    """几层 1D 卷积，回归每个音素的对数时长。"""

    def __init__(
        self, hidden: int, kernel_size: int = 3, n_layers: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers.append(nn.Conv1d(hidden, hidden, kernel_size, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Conv1d(hidden, 1, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``hidden`` 形状 (B, N, H)，返回对数时长 (B, N)。"""
        h = hidden.transpose(1, 2)
        h = self.convs(h)
        log_dur = self.proj(h).squeeze(1)
        if mask is not None:
            log_dur = log_dur * mask
        return log_dur


def regulate_length(mu: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按整数时长把逐音素条件展开到逐帧。

    参数
    ----
    mu : (B, C, N) 逐音素条件
    durations : (B, N) 每个音素占的帧数（会被下取整并截断到 >= 0）

    返回 (mu_frame, lengths)，其中 ``mu_frame`` 形状 (B, C, T_max)，右侧零填充。
    """
    batch, channels, _ = mu.shape
    dur = durations.long().clamp(min=0)
    lengths = dur.sum(dim=1)
    max_len = max(int(lengths.max().item()), 1) if batch else 1

    out = mu.new_zeros(batch, channels, max_len)
    for b in range(batch):
        expanded = torch.repeat_interleave(mu[b], dur[b], dim=1)
        out[b, :, : expanded.shape[1]] = expanded
    return out, lengths
