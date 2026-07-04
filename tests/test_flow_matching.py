"""条件流匹配模型的单元测试。"""

import torch

from voxflow.models.estimator import EstimatorConfig, VectorFieldEstimator
from voxflow.models.flow_matching import ConditionalFlowMatching


def _model():
    est = VectorFieldEstimator(
        EstimatorConfig(n_mels=16, hidden=16, spk_dim=8, time_dim=16, depth=2, heads=2)
    )
    return ConditionalFlowMatching(est, sigma_min=1e-4)


def test_loss_is_scalar_and_finite():
    cfm = _model()
    b, t = 2, 20
    x1 = torch.randn(b, 16, t)
    mu = torch.randn(b, 16, t)
    spk = torch.randn(b, 8)
    loss = cfm.compute_loss(x1, mu, spk)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_with_mask():
    cfm = _model()
    b, t = 2, 18
    mask = torch.ones(b, 1, t)
    mask[1, :, 10:] = 0
    loss = cfm.compute_loss(torch.randn(b, 16, t), torch.randn(b, 16, t), torch.randn(b, 8), mask)
    assert torch.isfinite(loss)


def test_sample_shape():
    cfm = _model()
    mu = torch.randn(1, 16, 22)
    spk = torch.randn(1, 8)
    mel = cfm.sample(mu, spk, n_timesteps=4)
    assert mel.shape == (1, 16, 22)
    assert torch.isfinite(mel).all()


def test_sample_rejects_zero_steps():
    cfm = _model()
    try:
        cfm.sample(torch.randn(1, 16, 10), torch.randn(1, 8), n_timesteps=0)
    except ValueError:
        return
    raise AssertionError("应当拒绝 n_timesteps=0")
