"""音频 I/O 与重采样的单元测试。"""

import numpy as np

from voxflow.audio.io import load_wav, peak_normalize, resample, save_wav, to_mono


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(4), np.zeros(4)], axis=1).astype(np.float32)
    mono = to_mono(stereo)
    assert mono.shape == (4,)
    assert np.allclose(mono, 0.5)


def test_peak_normalize():
    wav = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    out = peak_normalize(wav, peak=1.0)
    assert np.isclose(np.max(np.abs(out)), 1.0)


def test_peak_normalize_all_zero():
    wav = np.zeros(8, dtype=np.float32)
    assert np.array_equal(peak_normalize(wav), wav)


def test_resample_changes_length():
    wav = np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
    out = resample(wav, 16000, 8000)
    assert abs(len(out) - 8000) <= 2


def test_resample_noop():
    wav = np.ones(10, dtype=np.float32)
    assert np.array_equal(resample(wav, 22050, 22050), wav)


def test_wav_roundtrip(tmp_path):
    sr = 16000
    wav = (0.3 * np.sin(np.linspace(0, 20 * np.pi, sr))).astype(np.float32)
    path = tmp_path / "tone.wav"
    save_wav(path, wav, sr)
    loaded, loaded_sr = load_wav(path)
    assert loaded_sr == sr
    assert loaded.shape == wav.shape
    assert np.max(np.abs(loaded - wav)) < 1e-3
