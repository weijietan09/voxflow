"""mel 频谱提取的单元测试。"""

import torch

from voxflow.audio.mel import MelSpectrogram, mel_filterbank
from voxflow.config import AudioConfig


def test_filterbank_shape():
    fb = mel_filterbank(n_mels=80, n_fft=1024, sample_rate=22050, f_min=0.0, f_max=8000.0)
    assert fb.shape == (80, 513)
    assert torch.all(fb >= 0)


def test_mel_output_shape():
    cfg = AudioConfig()
    mel = MelSpectrogram(cfg)
    wav = torch.randn(cfg.sample_rate)  # 1 秒
    out = mel(wav)
    assert out.shape[0] == 1
    assert out.shape[1] == cfg.n_mels
    assert out.shape[2] > 0


def test_mel_batched():
    cfg = AudioConfig(sample_rate=16000, f_max=8000)
    mel = MelSpectrogram(cfg)
    wav = torch.randn(3, 16000)
    out = mel(wav)
    assert out.shape[0] == 3
    assert out.shape[1] == cfg.n_mels


def test_mel_is_finite():
    mel = MelSpectrogram()
    out = mel(torch.zeros(4096))
    assert torch.isfinite(out).all()
