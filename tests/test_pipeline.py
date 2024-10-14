"""端到端流水线的冒烟测试（随机初始化权重）。"""

import numpy as np
import torch

from voxflow.pipeline import PipelineConfig, VoiceCloner


def _small_cloner():
    # 用小配置让冒烟测试跑得快
    cfg = PipelineConfig()
    cfg.acoustic_hidden = 32
    cfg.text_hidden = 64
    cfg.n_timesteps = 2
    return VoiceCloner(cfg)


def test_embed_reference_from_array():
    cloner = _small_cloner()
    torch.manual_seed(0)
    wav = np.random.randn(16000).astype(np.float32) * 0.1
    emb = cloner.embed_reference(wav)
    assert emb.shape == (cloner.config.spk_dim,)
    assert torch.isclose(emb.norm(), torch.tensor(1.0), atol=1e-4)


def test_clone_end_to_end_produces_audio():
    cloner = _small_cloner()
    torch.manual_seed(0)
    ref = np.random.randn(16000).astype(np.float32) * 0.1
    wav = cloner.clone("你好world", ref)
    assert isinstance(wav, np.ndarray)
    assert wav.dtype == np.float32
    assert wav.ndim == 1 and wav.size > 0
    assert np.isfinite(wav).all()


def test_synthesize_mel_shape():
    cloner = _small_cloner()
    torch.manual_seed(0)
    spk = torch.randn(cloner.config.spk_dim)
    mel = cloner.synthesize_mel("测试", spk)
    assert mel.shape[0] == 1
    assert mel.shape[1] == cloner.config.audio.n_mels
