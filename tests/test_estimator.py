"""向量场估计器的单元测试。"""

import torch

from voxflow.models.estimator import EstimatorConfig, VectorFieldEstimator


def _tiny():
    return VectorFieldEstimator(
        EstimatorConfig(n_mels=16, hidden=16, spk_dim=8, time_dim=16, depth=2, heads=2)
    )


def test_estimator_output_shape():
    net = _tiny()
    b, t = 2, 20
    x = torch.randn(b, 16, t)
    mu = torch.randn(b, 16, t)
    spk = torch.randn(b, 8)
    ts = torch.rand(b)
    out = net(x, ts, mu, spk)
    assert out.shape == (b, 16, t)


def test_estimator_handles_odd_length():
    net = _tiny()
    b, t = 1, 15
    out = net(torch.randn(b, 16, t), torch.rand(b), torch.randn(b, 16, t), torch.randn(b, 8))
    assert out.shape == (b, 16, t)


def test_estimator_is_differentiable():
    net = _tiny()
    x = torch.randn(1, 16, 12, requires_grad=True)
    out = net(x, torch.rand(1), torch.randn(1, 16, 12), torch.randn(1, 8))
    out.sum().backward()
    assert x.grad is not None
