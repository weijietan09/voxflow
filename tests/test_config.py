"""针对配置对象的单元测试。"""

import pytest

from voxflow.config import AudioConfig, load_yaml
from voxflow.exceptions import ConfigError


def test_default_audio_config():
    cfg = AudioConfig()
    assert cfg.sample_rate == 22050
    assert cfg.n_mels == 80
    assert cfg.frame_shift_ms == pytest.approx(1000.0 * 256 / 22050)


def test_frames_for_is_monotonic():
    cfg = AudioConfig()
    assert cfg.frames_for(0) == 1
    assert cfg.frames_for(cfg.hop_length * 10) == 11


def test_invalid_hop_length_rejected():
    with pytest.raises(ConfigError):
        AudioConfig(hop_length=0)
    with pytest.raises(ConfigError):
        AudioConfig(hop_length=99999)


def test_fmax_above_nyquist_rejected():
    with pytest.raises(ConfigError):
        AudioConfig(sample_rate=16000, f_max=9000)


def test_from_dict_rejects_unknown_field():
    with pytest.raises(ConfigError):
        AudioConfig.from_dict({"sample_rate": 16000, "bogus": 1})


def test_roundtrip_dict():
    cfg = AudioConfig(sample_rate=16000, f_max=8000)
    assert AudioConfig.from_dict(cfg.to_dict()) == cfg


def test_load_yaml_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_yaml(tmp_path / "nope.yaml")
