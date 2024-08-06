"""GE2E 风格的说话人编码器（d-vector）。

结构参考 Wan et al. 2018 与 Resemblyzer：多层 LSTM 取末帧隐状态，
线性投影后 ReLU，再做 L2 归一化，得到单位球面上的说话人 embedding。
输入是说话人编码器专用的 log-mel（默认 16 kHz / 40 维）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SpeakerEncoderConfig:
    """说话人编码器超参数。"""

    mel_dim: int = 40
    hidden_size: int = 256
    num_layers: int = 3
    embedding_dim: int = 256


class SpeakerEncoder(nn.Module):
    """把一段（或一批）log-mel 编码成 L2 归一化的说话人 embedding。"""

    def __init__(self, config: SpeakerEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpeakerEncoderConfig()
        self.lstm = nn.LSTM(
            input_size=self.config.mel_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(self.config.hidden_size, self.config.embedding_dim)
        self.relu = nn.ReLU()

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """``mels`` 形状 (B, T, mel_dim)，返回 (B, embedding_dim) 的单位向量。"""
        outputs, _ = self.lstm(mels)
        last = outputs[:, -1]
        emb = self.relu(self.proj(last))
        return F.normalize(emb, p=2, dim=-1)
