"""损失函数与训练器的单元测试。"""

import torch

from voxflow.models.estimator import EstimatorConfig, VectorFieldEstimator
from voxflow.models.flow_matching import ConditionalFlowMatching
from voxflow.train.losses import MultiResolutionSTFTLoss, masked_l1_loss
from voxflow.train.trainer import TrainConfig, Trainer


def test_multi_resolution_stft_loss_finite():
    loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=(256, 128), hop_sizes=(64, 32), win_sizes=(256, 128)
    )
    pred = torch.randn(2, 4000)
    target = torch.randn(2, 4000)
    value = loss_fn(pred, target)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_masked_l1_ignores_padding():
    pred = torch.zeros(1, 4, 5)
    target = torch.ones(1, 4, 5)
    mask = torch.ones(1, 1, 5)
    mask[0, 0, 3:] = 0
    loss = masked_l1_loss(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_trainer_runs_and_updates_params():
    est = VectorFieldEstimator(
        EstimatorConfig(n_mels=16, hidden=16, spk_dim=8, time_dim=16, depth=2, heads=2)
    )
    model = ConditionalFlowMatching(est)

    def conditioner(batch):
        return batch["mel"], batch["mu"], batch["spk"], None

    trainer = Trainer(model, conditioner, TrainConfig(lr=1e-3, max_steps=3))
    before = next(model.parameters()).clone()
    batches = [
        {"mel": torch.randn(2, 16, 20), "mu": torch.randn(2, 16, 20), "spk": torch.randn(2, 8)}
        for _ in range(3)
    ]
    history = trainer.fit(batches)
    assert len(history) == 3
    assert all(torch.isfinite(torch.tensor(h)) for h in history)
    after = next(model.parameters())
    assert not torch.equal(before, after)
