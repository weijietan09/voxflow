"""极简训练循环。

:class:`Trainer` 只依赖模型暴露的 ``compute_loss``，具体如何从一个 batch
构造出它的输入由调用方传入的 ``conditioner`` 决定，这样训练器对模型内部保持无知。

注：目前尚未实现时长对齐（如 MAS），完整的端到端训练仍在推进中。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TrainConfig:
    """训练超参数。"""

    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_steps: int = 1000
    log_every: int = 50


ConditionFn = Callable[[dict], tuple[torch.Tensor, ...]]


class Trainer:
    """针对暴露了 ``compute_loss`` 的模型的通用训练器。

    ``model`` 需要是一个 ``nn.Module`` 且实现 ``compute_loss(*tensors) -> Tensor``
    （例如 :class:`~voxflow.models.flow_matching.ConditionalFlowMatching`）。
    """

    def __init__(
        self,
        model: nn.Module,
        conditioner: ConditionFn,
        config: TrainConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.conditioner = conditioner
        self.config = config or TrainConfig()
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.step = 0

    def train_step(self, batch: dict) -> float:
        self.model.train()
        args = self.conditioner(batch)
        loss = self.model.compute_loss(*args)  # type: ignore[operator]

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.step += 1
        return float(loss.detach())

    def fit(self, batches: Iterable[dict]) -> list[float]:
        """遍历 ``batches`` 训练，最多走 ``max_steps`` 步，返回每步损失。"""
        history: list[float] = []
        for batch in batches:
            if self.step >= self.config.max_steps:
                break
            history.append(self.train_step(batch))
        return history
