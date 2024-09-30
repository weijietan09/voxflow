"""声码器：把 mel 频谱转回波形。

提供两种实现：

- :class:`HiFiGANLite`：小号 HiFi-GAN 生成器（转置卷积上采样 + 多感受野残差
  块），结构可训练，权重需自行训练；
- :class:`GriffinLimVocoder`：无需训练的相位重建后备方案，便于早期打通链路。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from voxflow.audio.mel import mel_filterbank
from voxflow.config import AudioConfig


def _get_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size - 1) * dilation // 2


class ResBlock(nn.Module):
    """HiFi-GAN 的多膨胀残差块。"""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: Sequence[int] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=_get_padding(kernel_size, d))
                for d in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=_get_padding(kernel_size, 1))
                for _ in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            xt = conv1(F.leaky_relu(x, 0.1))
            xt = conv2(F.leaky_relu(xt, 0.1))
            x = x + xt
        return x


class HiFiGANLite(nn.Module):
    """精简版 HiFi-GAN 生成器。

    上采样倍率之积应等于声学特征的 hop_length，这样输出采样点数正好是
    ``帧数 × hop_length``。
    """

    def __init__(
        self,
        n_mels: int = 80,
        base_channels: int = 128,
        upsample_rates: Sequence[int] = (8, 8, 4),
        upsample_kernel_sizes: Sequence[int] = (16, 16, 8),
        resblock_kernels: Sequence[int] = (3, 7, 11),
        resblock_dilations: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernels)
        self.conv_pre = nn.Conv1d(n_mels, base_channels, 7, padding=3)

        self.ups = nn.ModuleList()
        channels = base_channels
        for rate, kernel in zip(upsample_rates, upsample_kernel_sizes, strict=True):
            self.ups.append(
                nn.ConvTranspose1d(channels, channels // 2, kernel, stride=rate, padding=(kernel - rate) // 2)
            )
            channels //= 2

        self.resblocks = nn.ModuleList()
        channels = base_channels
        for _ in self.ups:
            channels //= 2
            for kernel, dilations in zip(resblock_kernels, resblock_dilations, strict=True):
                self.resblocks.append(ResBlock(channels, kernel, dilations))

        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """``mel`` 形状 (B, n_mels, T)，返回 (B, 1, T × ∏upsample_rates)。"""
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, 0.1))
            acc = None
            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x)
                acc = out if acc is None else acc + out
            x = acc / self.num_kernels
        x = self.conv_post(F.leaky_relu(x, 0.1))
        return torch.tanh(x)


class GriffinLimVocoder(nn.Module):
    """基于 Griffin-Lim 的免训练声码器。

    先用 mel 滤波器组的伪逆把 log-mel 近似还原成线性幅度谱，再用若干轮
    Griffin-Lim 迭代恢复相位。音质有限，但胜在不需要权重，方便早期验证链路。
    """

    def __init__(self, config: AudioConfig | None = None, n_iters: int = 32) -> None:
        super().__init__()
        self.config = config or AudioConfig()
        self.n_iters = n_iters
        fb = mel_filterbank(
            self.config.n_mels,
            self.config.n_fft,
            self.config.sample_rate,
            self.config.f_min,
            self.config.f_max,
        )
        self.register_buffer("inv_filterbank", torch.linalg.pinv(fb))

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """``log_mel`` 形状 (B, n_mels, T)，返回波形 (B, samples)。"""
        cfg = self.config
        mel_amp = torch.exp(log_mel)
        linear = torch.matmul(self.inv_filterbank.to(mel_amp.dtype), mel_amp).clamp(min=1e-5)

        window = torch.hann_window(cfg.win_length, device=log_mel.device, dtype=log_mel.dtype)
        angle = 2.0 * torch.pi * torch.rand_like(linear)
        phase = torch.polar(torch.ones_like(linear), angle)

        wav = None
        for _ in range(self.n_iters):
            spec = linear * phase
            wav = torch.istft(spec, cfg.n_fft, cfg.hop_length, cfg.win_length, window=window)
            rebuilt = torch.stft(
                wav,
                cfg.n_fft,
                cfg.hop_length,
                cfg.win_length,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            phase = rebuilt / (rebuilt.abs() + 1e-8)

        spec = linear * phase
        return torch.istft(spec, cfg.n_fft, cfg.hop_length, cfg.win_length, window=window)

