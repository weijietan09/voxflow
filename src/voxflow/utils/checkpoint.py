"""模型权重的保存与加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from voxflow.exceptions import CheckpointError


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """把模型（可选还有优化器状态）保存到 ``path``。"""
    payload: dict[str, Any] = {"model": model.state_dict(), "step": int(step)}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """把 ``path`` 的权重载入 ``model``，返回完整的 payload。"""
    file = Path(path)
    if not file.is_file():
        raise CheckpointError(f"找不到权重文件: {file}")
    payload = torch.load(str(file), map_location=map_location, weights_only=False)
    if "model" not in payload:
        raise CheckpointError("权重文件缺少 'model' 字段")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
