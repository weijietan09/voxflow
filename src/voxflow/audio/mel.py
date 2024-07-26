"""log-mel 频谱提取（torch 实现），参数由 :class:`AudioConfig` 给定。"""

from __future__ import annotations

import torch

from voxflow.audio.stft import magnitude
from voxflow.config import AudioConfig


def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    """HTK 约定下的 Hz -> mel。"""
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """HTK 约定下的 mel -> Hz。"""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    """构造三角 mel 滤波器组，形状 ``(n_mels, n_fft // 2 + 1)``。"""
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)

    m_min = hz_to_mel(torch.tensor(f_min))
    m_max = hz_to_mel(torch.tensor(f_max))
    m_points = torch.linspace(float(m_min), float(m_max), n_mels + 2)
    f_points = mel_to_hz(m_points)

    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        lower, center, upper = f_points[i], f_points[i + 1], f_points[i + 2]
        left_slope = (all_freqs - lower) / (center - lower)
        right_slope = (upper - all_freqs) / (upper - center)
        fb[i] = torch.clamp(torch.minimum(left_slope, right_slope), min=0.0)
    return fb


class MelSpectrogram(torch.nn.Module):
    """把波形转成 log-mel 频谱的模块。

    滤波器组作为 buffer 注册，随模型一起 ``.to(device)``。
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        super().__init__()
        self.config = config or AudioConfig()
        fb = mel_filterbank(
            self.config.n_mels,
            self.config.n_fft,
            self.config.sample_rate,
            self.config.f_min,
            self.config.f_max,
        )
        self.register_buffer("filterbank", fb)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """``wav`` 形状 (T,) 或 (B, T)，返回 (B, n_mels, frames)。"""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mag = magnitude(
            wav,
            self.config.n_fft,
            self.config.hop_length,
            self.config.win_length,
        )
        mel = torch.matmul(self.filterbank.to(mag.dtype), mag)
        return torch.log(torch.clamp(mel, min=self.config.mel_floor))
