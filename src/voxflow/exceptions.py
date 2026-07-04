"""VoxFlow 的异常类型层级。

对外统一从 :class:`VoxFlowError` 派生，调用方可以只捕获基类，
也可以针对具体阶段（配置 / 音频 / 文本 / 权重）分别处理。
"""

from __future__ import annotations


class VoxFlowError(Exception):
    """所有 VoxFlow 异常的基类。"""


class ConfigError(VoxFlowError):
    """配置缺失、类型不匹配或取值非法时抛出。"""


class AudioError(VoxFlowError):
    """音频读写、重采样或特征提取失败时抛出。"""


class TextFrontendError(VoxFlowError):
    """文本前端无法处理输入时抛出。"""


class CheckpointError(VoxFlowError):
    """模型权重加载 / 保存失败时抛出。"""
