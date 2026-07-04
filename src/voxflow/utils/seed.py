"""可复现实验用的随机种子设置。

对 ``random`` / ``numpy`` / ``torch`` 统一播种。numpy 与 torch 为可选
导入——只做文本处理时即使没装这两个库也能正常工作。
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 1234, deterministic: bool = False) -> int:
    """设置各随机源的种子，返回所用的 ``seed`` 方便记录。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy 属核心依赖，通常都在
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass

    return seed
