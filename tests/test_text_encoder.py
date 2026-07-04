"""文本编码器与时长模块的单元测试。"""

import torch

from voxflow.models.duration import DurationPredictor, regulate_length
from voxflow.models.text_encoder import TextEncoder, TextEncoderConfig


def test_text_encoder_shapes():
    enc = TextEncoder(TextEncoderConfig(vocab_size=50, hidden=32, n_mels=16, n_layers=2, heads=4))
    ids = torch.randint(0, 50, (2, 7))
    mu, hidden = enc(ids)
    assert mu.shape == (2, 16, 7)
    assert hidden.shape == (2, 7, 32)


def test_duration_predictor_shape():
    dp = DurationPredictor(hidden=32)
    hidden = torch.randn(2, 7, 32)
    log_dur = dp(hidden)
    assert log_dur.shape == (2, 7)


def test_regulate_length_expands():
    mu = torch.randn(1, 4, 3)
    durations = torch.tensor([[2, 1, 3]])
    frames, lengths = regulate_length(mu, durations)
    assert frames.shape == (1, 4, 6)
    assert lengths.tolist() == [6]
    # 第一个音素被复制两帧
    assert torch.allclose(frames[0, :, 0], frames[0, :, 1])


def test_regulate_length_pads_batch():
    mu = torch.randn(2, 4, 2)
    durations = torch.tensor([[1, 1], [3, 2]])
    frames, lengths = regulate_length(mu, durations)
    assert frames.shape == (2, 4, 5)
    assert lengths.tolist() == [2, 5]
