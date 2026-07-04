"""条件流匹配（Conditional Flow Matching）声学模型。

训练目标（最优传输 CFM）：采样 t ~ U(0, 1)、噪声 ``x0 ~ N(0, I)``，
构造从噪声到目标 mel ``x1`` 的直线插值

    x_t = (1 - (1 - σ_min) t) · x0 + t · x1,

其对应的目标向量场为 ``u = x1 - (1 - σ_min) x0``，
用估计器预测 ``v(x_t, t, 条件)`` 与之做 MSE。

采样：从高斯噪声出发，用欧拉法沿 t: 0 → 1 积分 ``dx/dt = v``。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from voxflow.models.estimator import VectorFieldEstimator


class ConditionalFlowMatching(nn.Module):
    """把向量场估计器包成一个可训练 / 可采样的 CFM 声学模型。"""

    def __init__(self, estimator: VectorFieldEstimator, sigma_min: float = 1e-4) -> None:
        super().__init__()
        self.estimator = estimator
        self.sigma_min = sigma_min

    @property
    def n_mels(self) -> int:
        return self.estimator.config.n_mels

    def compute_loss(
        self,
        x1: torch.Tensor,
        mu: torch.Tensor,
        spk: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """给定目标 mel ``x1`` 与条件，返回 CFM 的 MSE 损失。

        ``mask`` 形状 (B, 1, T)，用于在变长 batch 里忽略 padding 帧。
        """
        batch = x1.shape[0]
        t = torch.rand(batch, device=x1.device, dtype=x1.dtype)
        x0 = torch.randn_like(x1)
        t_expand = t.view(batch, 1, 1)

        x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x0 + t_expand * x1
        target = x1 - (1.0 - self.sigma_min) * x0
        pred = self.estimator(x_t, t, mu, spk)

        if mask is None:
            return F.mse_loss(pred, target)
        diff = (pred - target) * mask
        denom = mask.sum() * x1.shape[1] + 1e-8
        return diff.pow(2).sum() / denom

    @torch.inference_mode()
    def sample(
        self,
        mu: torch.Tensor,
        spk: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """从噪声出发用欧拉法积分，生成 mel，形状与 ``mu`` 相同。

        ``n_timesteps`` 越大越接近 ODE 精确解，但推理越慢；
        流匹配的一大好处正是很少的步数（个位数）也能出可用结果。
        """
        if n_timesteps < 1:
            raise ValueError("n_timesteps 必须 >= 1")
        batch, _, length = mu.shape
        x = torch.randn(batch, self.n_mels, length, device=mu.device, dtype=mu.dtype)
        x = x * temperature

        t_span = torch.linspace(0.0, 1.0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        for i in range(n_timesteps):
            t_cur = t_span[i].expand(batch)
            dt = t_span[i + 1] - t_span[i]
            velocity = self.estimator(x, t_cur, mu, spk)
            x = x + dt * velocity
        return x
