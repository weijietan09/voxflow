"""VoxFlow —— 零样本声音克隆 TTS 工具包。

顶层只保留版本号是即时可用的；其余重量级对象（依赖 torch）通过
模块级 ``__getattr__`` 懒加载，这样仅做文本处理时无需引入 torch。
"""

from __future__ import annotations

import importlib
from typing import Any

from voxflow.version import __version__

# 公开对象 -> 所在模块，按需懒加载
_LAZY: dict[str, str] = {
    "VoiceCloner": "voxflow.pipeline",
    "PipelineConfig": "voxflow.pipeline",
    "TextFrontend": "voxflow.text.frontend",
    "AudioConfig": "voxflow.config",
    "MelSpectrogram": "voxflow.audio.mel",
    "SpeakerEncoder": "voxflow.models.speaker_encoder",
    "ConditionalFlowMatching": "voxflow.models.flow_matching",
    "GaussianDiffusion": "voxflow.models.diffusion",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module 'voxflow' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
