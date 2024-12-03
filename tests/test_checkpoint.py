"""权重保存 / 加载的单元测试。"""

import pytest
import torch
from torch import nn

from voxflow.exceptions import CheckpointError
from voxflow.utils.checkpoint import load_checkpoint, save_checkpoint


def test_roundtrip_preserves_weights(tmp_path):
    src = nn.Linear(4, 3)
    optimizer = torch.optim.Adam(src.parameters())
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, src, optimizer, step=7, extra={"note": "hi"})

    dst = nn.Linear(4, 3)
    payload = load_checkpoint(path, dst)
    assert payload["step"] == 7
    assert payload["extra"]["note"] == "hi"
    assert torch.equal(src.weight, dst.weight)


def test_missing_file_raises(tmp_path):
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "absent.pt", nn.Linear(2, 2))
