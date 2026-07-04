"""训练用损失函数。

- :class:`MultiResolutionSTFTLoss`：多分辨率 STFT 损失，训练声码器时用；
- :func:`masked_l1_loss`：支持 padding 掩码的 mel L1 损失。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _stft_magnitude(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x, n_fft, hop, win, window=window, center=True, pad_mode="reflect", return_complex=True
    )
    return spec.abs().clamp(min=1e-7)


class STFTLoss(nn.Module):
    """单一分辨率下的谱收敛 + 对数幅度损失。"""

    def __init__(self, n_fft: int = 1024, hop: int = 256, win: int = 1024) -> None:
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_mag = _stft_magnitude(pred, self.n_fft, self.hop, self.win)
        target_mag = _stft_magnitude(target, self.n_fft, self.hop, self.win)
        sc = torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-7)
        mag = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))
        return sc, mag


class MultiResolutionSTFTLoss(nn.Module):
    """多个 STFT 分辨率下损失的平均。"""

    def __init__(
        self,
        fft_sizes: Sequence[int] = (1024, 512, 2048),
        hop_sizes: Sequence[int] = (256, 128, 512),
        win_sizes: Sequence[int] = (1024, 512, 2048),
    ) -> None:
        super().__init__()
        self.losses = nn.ModuleList(
            [STFTLoss(n, h, w) for n, h, w in zip(fft_sizes, hop_sizes, win_sizes, strict=True)]
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sc_total = pred.new_zeros(())
        mag_total = pred.new_zeros(())
        for loss in self.losses:
            sc, mag = loss(pred, target)
            sc_total = sc_total + sc
            mag_total = mag_total + mag
        n = len(self.losses)
        return (sc_total + mag_total) / n


def masked_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """mel L1 损失，``mask`` 形状 (B, 1, T) 时忽略 padding 帧。"""
    if mask is None:
        return F.l1_loss(pred, target)
    diff = (pred - target).abs() * mask
    return diff.sum() / (mask.sum() * pred.shape[1] + 1e-8)
