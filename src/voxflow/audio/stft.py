"""基于 :func:`torch.stft` 的短时傅里叶变换封装。"""

from __future__ import annotations

import torch


def stft(
    wav: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
) -> torch.Tensor:
    """计算复数 STFT，返回形状 ``(..., n_fft // 2 + 1, frames)``。"""
    window = torch.hann_window(win_length, device=wav.device, dtype=wav.dtype)
    return torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode="reflect",
        return_complex=True,
    )


def magnitude(
    wav: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
) -> torch.Tensor:
    """STFT 幅度谱。"""
    return stft(wav, n_fft, hop_length, win_length, center=center).abs()
