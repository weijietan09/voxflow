"""特征与音频相关的配置对象。

这里刻意只用 ``dataclass`` 加一个轻量的 YAML 读取函数，
不引入 Hydra / OmegaConf 这类较重的依赖，方便在离线环境里复现。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from voxflow.exceptions import ConfigError


@dataclass(frozen=True)
class AudioConfig:
    """声学特征提取参数。

    默认值对应 22.05 kHz、80 维 log-mel，与常见的 TTS 声学模型 /
    声码器训练设置保持一致。说话人编码器另有一套 16 kHz / 40 维的配置。
    """

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    mel_floor: float = 1e-5

    def __post_init__(self) -> None:
        if self.hop_length <= 0 or self.hop_length > self.n_fft:
            raise ConfigError("hop_length 必须落在 (0, n_fft] 区间内")
        if self.win_length > self.n_fft:
            raise ConfigError("win_length 不能大于 n_fft")
        if self.f_max <= self.f_min:
            raise ConfigError("f_max 必须大于 f_min")
        if self.f_max > self.sample_rate / 2:
            raise ConfigError("f_max 不能超过奈奎斯特频率 (sample_rate / 2)")

    @property
    def frame_shift_ms(self) -> float:
        """相邻两帧之间的时间间隔（毫秒）。"""
        return 1000.0 * self.hop_length / self.sample_rate

    def frames_for(self, num_samples: int) -> int:
        """给定波形采样点数，估算 mel 帧数（中心填充约定）。"""
        return 1 + num_samples // self.hop_length

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"AudioConfig 收到未知字段: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AudioConfig:
        payload = load_yaml(path)
        section = payload.get("audio", payload)
        return cls.from_dict(section)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取一个 YAML 文件并返回字典，出错时抛出 :class:`ConfigError`。"""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"找不到配置文件: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - 依赖底层解析器
        raise ConfigError(f"解析 YAML 失败: {p}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射: {p}")
    return data
