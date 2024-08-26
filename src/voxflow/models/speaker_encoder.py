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

    @torch.inference_mode()
    def embed_utterance(
        self,
        mel: torch.Tensor,
        partial_frames: int = 160,
        hop_frames: int = 80,
    ) -> torch.Tensor:
        """对整段音频做 partial-utterance 平均，得到单条 embedding。

        把 ``mel``（形状 (T, mel_dim)）切成若干等长的重叠片段分别编码，
        再对结果取平均并重新 L2 归一化。片段不足一个窗口时直接整段编码。
        """
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        slices = compute_partial_slices(mel.shape[0], partial_frames, hop_frames)
        if len(slices) == 1:
            start, end = slices[0]
            partials = mel[start:end].unsqueeze(0)
        else:
            partials = torch.stack([mel[start:end] for start, end in slices])
        embeds = self.forward(partials)
        mean = embeds.mean(dim=0)
        return F.normalize(mean, p=2, dim=-1)


def compute_partial_slices(
    n_frames: int, partial_frames: int = 160, hop_frames: int = 80
) -> list[tuple[int, int]]:
    """把 ``n_frames`` 切成等长重叠片段的 (起, 止) 索引；不足一窗则整段返回。"""
    slices: list[tuple[int, int]] = []
    start = 0
    while start + partial_frames <= n_frames:
        slices.append((start, start + partial_frames))
        start += hop_frames
    if not slices:
        slices.append((0, n_frames))
    return slices
