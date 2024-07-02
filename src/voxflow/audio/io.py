"""波形读写、单声道化与幅度归一化。

用 ``soundfile`` 做 I/O（wheel 自带 libsndfile，离线可用），
统一以 float32、[-1, 1] 的单声道 numpy 数组在内部流转。
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from voxflow.exceptions import AudioError


def to_mono(wav: np.ndarray) -> np.ndarray:
    """把多声道波形按通道求平均降为单声道。"""
    if wav.ndim == 1:
        return wav
    if wav.ndim == 2:
        return wav.mean(axis=1)
    raise AudioError(f"波形维度不受支持: {wav.ndim}")


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """多相（polyphase）重采样，采样率相同时直接返回。"""
    if orig_sr == target_sr:
        return wav.astype(np.float32)
    if orig_sr <= 0 or target_sr <= 0:
        raise AudioError("采样率必须为正整数")
    divisor = gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    return resample_poly(wav, up, down).astype(np.float32)


def peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """按峰值把波形缩放到 ``peak``，全零输入原样返回。"""
    max_abs = float(np.max(np.abs(wav))) if wav.size else 0.0
    if max_abs <= 1e-8:
        return wav
    return (wav / max_abs * peak).astype(np.float32)


def load_wav(path: str | Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """读取音频文件，返回 (单声道 float32 波形, 采样率)。

    给定 ``sample_rate`` 时会重采样到该采样率。
    """
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except (RuntimeError, OSError) as exc:
        raise AudioError(f"读取音频失败: {path}: {exc}") from exc
    wav = to_mono(np.asarray(wav, dtype=np.float32))
    if sample_rate is not None and sr != sample_rate:
        wav = resample(wav, int(sr), sample_rate)
        sr = sample_rate
    return wav, int(sr)


def save_wav(path: str | Path, wav: np.ndarray, sample_rate: int) -> None:
    """把单声道波形写成 WAV 文件。"""
    wav = np.asarray(wav, dtype=np.float32)
    try:
        sf.write(str(path), wav, sample_rate, subtype="PCM_16")
    except (RuntimeError, OSError) as exc:
        raise AudioError(f"写入音频失败: {path}: {exc}") from exc
