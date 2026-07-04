"""共享的 pytest fixtures。"""

import numpy as np
import pytest

from voxflow.audio.io import save_wav


@pytest.fixture
def make_wav(tmp_path):
    """生成一个正弦波 wav 文件并返回其路径。"""

    def _make(name: str = "sample.wav", seconds: float = 1.0, sample_rate: int = 22050):
        n = int(seconds * sample_rate)
        wav = (0.1 * np.sin(np.linspace(0.0, 60.0 * np.pi, n))).astype(np.float32)
        path = tmp_path / name
        save_wav(path, wav, sample_rate)
        return path

    return _make
