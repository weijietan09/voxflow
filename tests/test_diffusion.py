"""DDPM 扩散模型的单元测试。"""

import torch

from voxflow.models.diffusion import GaussianDiffusion, linear_beta_schedule
from voxflow.models.estimator import EstimatorConfig, VectorFieldEstimator


def _model(n_steps=20):
    est = VectorFieldEstimator(
        EstimatorConfig(n_mels=16, hidden=16, spk_dim=8, time_dim=16, depth=2, heads=2)
    )
    return GaussianDiffusion(est, n_steps=n_steps)


def test_beta_schedule_monotonic():
    betas = linear_beta_schedule(50)
    assert betas[0] < betas[-1]
    assert torch.all(betas > 0)


def test_q_sample_shape():
    diff = _model()
    x0 = torch.randn(2, 16, 12)
    t = torch.tensor([0, 5])
    out = diff.q_sample(x0, t, torch.randn_like(x0))
    assert out.shape == x0.shape


def test_loss_finite():
    diff = _model()
    loss = diff.compute_loss(torch.randn(2, 16, 14), torch.randn(2, 16, 14), torch.randn(2, 8))
    assert torch.isfinite(loss)


def test_reverse_sampling_shape():
    diff = _model(n_steps=10)
    mu = torch.randn(1, 16, 12)
    mel = diff.sample(mu, torch.randn(1, 8))
    assert mel.shape == (1, 16, 12)
    assert torch.isfinite(mel).all()
