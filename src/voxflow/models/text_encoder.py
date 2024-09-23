"""文本编码器：音素 id 序列 -> Transformer -> 逐音素条件均值 μ。

输出两部分：
- ``mu``：投影到 mel 维度的逐音素均值，长度对齐后作为声学模型的条件；
- ``hidden``：Transformer 隐状态，供时长预测器使用。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from voxflow.models.modules import SinusoidalPosEmb, TransformerBlock


@dataclass
class TextEncoderConfig:
    """文本编码器超参数。"""

    vocab_size: int
    hidden: int = 192
    n_mels: int = 80
    n_layers: int = 4
    heads: int = 4


class TextEncoder(nn.Module):
    """把音素 id 编码成逐音素的条件均值与隐状态。"""

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden, padding_idx=0)
        self.pos_emb = SinusoidalPosEmb(config.hidden)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config.hidden, heads=config.heads) for _ in range(config.n_layers)]
        )
        self.to_mu = nn.Linear(config.hidden, config.n_mels)

    def forward(
        self, ids: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``ids`` 形状 (B, N)，返回 (mu, hidden)。

        - ``mu``：(B, n_mels, N)
        - ``hidden``：(B, N, hidden)
        """
        _, n = ids.shape
        x = self.embed(ids)
        positions = self.pos_emb(torch.arange(n, device=ids.device))
        x = x + positions.unsqueeze(0)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        mu = self.to_mu(x).transpose(1, 2)
        return mu, x
