"""示例 03：在合成数据上训练一个很小的条件流匹配模型。

构造一个玩具任务——让模型学会从随机条件 μ 生成"目标 mel"（这里直接
用 μ 加一点噪声当目标），训练几十步观察损失下降，再采样一次看形状。
纯粹用于验证训练 / 采样链路，不追求音质。
"""

import torch

from voxflow.models.flow_matching import ConditionalFlowMatching
from voxflow.models.velocity_field import EstimatorConfig, VectorFieldEstimator
from voxflow.train.trainer import TrainConfig, Trainer


def main() -> None:
    torch.manual_seed(0)
    n_mels, length = 16, 24

    model = ConditionalFlowMatching(
        VectorFieldEstimator(
            EstimatorConfig(n_mels=n_mels, hidden=32, spk_dim=8, time_dim=16, depth=2, heads=2)
        )
    )

    def conditioner(batch):
        return batch["x1"], batch["mu"], batch["spk"], None

    trainer = Trainer(model, conditioner, TrainConfig(lr=1e-3, max_steps=40))

    def make_batch():
        mu = torch.randn(4, n_mels, length)
        x1 = mu + 0.1 * torch.randn_like(mu)  # 目标与条件强相关
        return {"x1": x1, "mu": mu, "spk": torch.randn(4, 8)}

    history = trainer.fit(make_batch() for _ in range(40))
    print(f"首步损失: {history[0]:.4f} -> 末步损失: {history[-1]:.4f}")

    mu = torch.randn(1, n_mels, length)
    mel = model.sample(mu, torch.randn(1, 8), n_timesteps=16)
    print(f"采样得到的 mel 形状: {tuple(mel.shape)}")


if __name__ == "__main__":
    main()
