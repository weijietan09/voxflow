"""模型通用组件的单元测试。"""

import torch

from voxflow.models.modules import (
    ResnetBlock1D,
    SinusoidalPosEmb,
    TransformerBlock,
    sequence_mask,
)


def test_sequence_mask():
    mask = sequence_mask(torch.tensor([2, 4]), max_len=4)
    assert mask.shape == (2, 4)
    assert mask[0].tolist() == [True, True, False, False]


def test_sinusoidal_shape():
    emb = SinusoidalPosEmb(16)
    out = emb(torch.arange(5))
    assert out.shape == (5, 16)


def test_resnet_block_shape():
    block = ResnetBlock1D(in_ch=8, out_ch=16, time_dim=32)
    x = torch.randn(2, 8, 10)
    t = torch.randn(2, 32)
    out = block(x, t)
    assert out.shape == (2, 16, 10)


def test_transformer_block_preserves_shape():
    block = TransformerBlock(dim=32, heads=4)
    x = torch.randn(2, 7, 32)
    assert block(x).shape == (2, 7, 32)


def test_transformer_respects_padding_mask():
    block = TransformerBlock(dim=16, heads=2)
    x = torch.randn(1, 5, 16)
    mask = torch.tensor([[True, True, True, False, False]])
    out = block(x, key_padding_mask=mask)
    assert torch.isfinite(out).all()
