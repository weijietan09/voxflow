"""声码器的单元测试。"""

import torch

from voxflow.config import AudioConfig
from voxflow.models.vocoder import GriffinLimVocoder, HiFiGANLite


def test_hifigan_upsamples_to_hop_length():
    voc = HiFiGANLite(
        n_mels=16, base_channels=16, upsample_rates=(4, 4), upsample_kernel_sizes=(8, 8)
    )
    mel = torch.randn(1, 16, 10)
    wav = voc(mel)
    assert wav.shape == (1, 1, 10 * 16)
    assert torch.all(wav.abs() <= 1.0)


def test_hifigan_batched():
    voc = HiFiGANLite(
        n_mels=16, base_channels=16, upsample_rates=(2, 2), upsample_kernel_sizes=(4, 4)
    )
    wav = voc(torch.randn(3, 16, 8))
    assert wav.shape[0] == 3
    assert torch.isfinite(wav).all()


def test_griffin_lim_produces_waveform():
    cfg = AudioConfig(
        sample_rate=16000, n_fft=256, hop_length=64, win_length=256, n_mels=32, f_max=8000
    )
    voc = GriffinLimVocoder(cfg, n_iters=4)
    log_mel = torch.randn(1, 32, 12) * 0.1
    wav = voc(log_mel)
    assert wav.ndim == 2
    assert wav.shape[1] > 0
    assert torch.isfinite(wav).all()
