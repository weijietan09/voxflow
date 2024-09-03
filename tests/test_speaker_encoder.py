"""说话人编码器的单元测试。"""

import torch

from voxflow.models.speaker_encoder import (
    SpeakerEncoder,
    SpeakerEncoderConfig,
    compute_partial_slices,
)


def test_forward_is_unit_norm():
    enc = SpeakerEncoder(SpeakerEncoderConfig(mel_dim=40, hidden_size=32, num_layers=2))
    mels = torch.randn(4, 50, 40)
    emb = enc(mels)
    assert emb.shape == (4, 256)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_compute_partial_slices_overlap():
    slices = compute_partial_slices(400, partial_frames=160, hop_frames=80)
    assert slices[0] == (0, 160)
    assert all(e - s == 160 for s, e in slices)


def test_compute_partial_slices_short_input():
    assert compute_partial_slices(50, partial_frames=160) == [(0, 50)]


def test_embed_utterance_unit_norm():
    enc = SpeakerEncoder(SpeakerEncoderConfig(mel_dim=40, hidden_size=32, num_layers=1))
    mel = torch.randn(500, 40)
    emb = enc.embed_utterance(mel)
    assert emb.shape == (256,)
    assert torch.isclose(emb.norm(), torch.tensor(1.0), atol=1e-5)
