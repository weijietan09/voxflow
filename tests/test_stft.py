"""STFT 封装的单元测试。"""

import torch

from voxflow.audio.stft import magnitude, stft


def test_stft_complex_shape():
    wav = torch.randn(2, 8000)
    spec = stft(wav, n_fft=1024, hop_length=256, win_length=1024)
    assert spec.shape[0] == 2
    assert spec.shape[1] == 513
    assert spec.is_complex()


def test_magnitude_nonnegative():
    wav = torch.randn(4000)
    mag = magnitude(wav, n_fft=512, hop_length=128, win_length=512)
    assert torch.all(mag >= 0)
