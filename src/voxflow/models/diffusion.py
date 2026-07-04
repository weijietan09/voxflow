"""DDPM 扩散声学模型（流匹配之外的对照实现）。

复用同一个 :class:`VectorFieldEstimator`，把它当作噪声预测网络 ε_θ。
提供线性 β 调度、前向加噪 q(x_t | x0) 与 DDPM 祖先采样。留着它一方面便于
和流匹配做同条件下的对比，另一方面也验证估计器接口对两类目标都够通用。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from voxflow.models.estimator import VectorFieldEstimator


def linear_beta_schedule(n_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """线性 β 调度。"""
    return torch.linspace(beta_start, beta_end, n_steps)


class GaussianDiffusion(nn.Module):
    """标准 DDPM，噪声预测参数化。"""

    def __init__(
        self,
        estimator: VectorFieldEstimator,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self.n_steps = n_steps

        betas = linear_beta_schedule(n_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_acp", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - alphas_cumprod))

    @property
    def n_mels(self) -> int:
        return self.estimator.config.n_mels

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """前向扩散：给 ``x0`` 在第 ``t`` 步加噪。"""
        sqrt_acp = self.sqrt_acp[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_acp[t].view(-1, 1, 1)
        return sqrt_acp * x0 + sqrt_om * noise

    def compute_loss(
        self,
        x1: torch.Tensor,
        mu: torch.Tensor,
        spk: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """噪声预测的 MSE 损失。"""
        batch = x1.shape[0]
        t = torch.randint(0, self.n_steps, (batch,), device=x1.device)
        noise = torch.randn_like(x1)
        x_t = self.q_sample(x1, t, noise)
        pred = self.estimator(x_t, t.to(x1.dtype) / self.n_steps, mu, spk)

        if mask is None:
            return F.mse_loss(pred, noise)
        diff = (pred - noise) * mask
        return diff.pow(2).sum() / (mask.sum() * x1.shape[1] + 1e-8)

    @torch.inference_mode()
    def sample(self, mu: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """DDPM 祖先采样，反向走完全部 ``n_steps``。"""
        batch, _, length = mu.shape
        x = torch.randn(batch, self.n_mels, length, device=mu.device, dtype=mu.dtype)

        for i in reversed(range(self.n_steps)):
            t = torch.full((batch,), i, device=mu.device, dtype=torch.long)
            eps = self.estimator(x, t.to(mu.dtype) / self.n_steps, mu, spk)
            beta = self.betas[i]
            alpha = self.alphas[i]
            coef = beta / self.sqrt_one_minus_acp[i]
            mean = (x - coef * eps) / torch.sqrt(alpha)
            if i > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x
