"""高层零样本声音克隆流水线。

把文本前端、说话人编码器、文本编码器 + 时长、声学模型（CFM / 扩散）与声码器
串成一个对象。默认权重是随机初始化的——本仓库当前聚焦可复现的框架与推理链路，
训练脚本见 :mod:`voxflow.train`，预训练权重需自行提供。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from voxflow.audio.io import load_wav, peak_normalize
from voxflow.audio.mel import MelSpectrogram
from voxflow.config import AudioConfig
from voxflow.models.duration import DurationPredictor, regulate_length
from voxflow.models.estimator import EstimatorConfig, VectorFieldEstimator
from voxflow.models.flow_matching import ConditionalFlowMatching
from voxflow.models.speaker_encoder import SpeakerEncoder, SpeakerEncoderConfig
from voxflow.models.text_encoder import TextEncoder, TextEncoderConfig
from voxflow.models.vocoder import GriffinLimVocoder
from voxflow.text.frontend import TextFrontend
from voxflow.text.symbols import DEFAULT_TABLE, SymbolTable


def _speaker_mel_config() -> AudioConfig:
    """说话人编码器专用的 16 kHz / 40 维 mel 配置。"""
    return AudioConfig(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=40,
        f_max=8000.0,
    )


@dataclass
class PipelineConfig:
    """流水线各子模块的尺寸配置。"""

    audio: AudioConfig = field(default_factory=AudioConfig)
    speaker_mel: AudioConfig = field(default_factory=_speaker_mel_config)
    spk_dim: int = 256
    text_hidden: int = 192
    acoustic_hidden: int = 128
    n_timesteps: int = 10
    max_frames_per_token: int = 50


class VoiceCloner(nn.Module):
    """端到端的零样本声音克隆器。"""

    def __init__(
        self, config: PipelineConfig | None = None, table: SymbolTable = DEFAULT_TABLE
    ) -> None:
        super().__init__()
        self.config = config or PipelineConfig()
        cfg = self.config
        self.table = table
        self.frontend = TextFrontend(table)

        self.speaker_mel = MelSpectrogram(cfg.speaker_mel)
        self.speaker_encoder = SpeakerEncoder(
            SpeakerEncoderConfig(mel_dim=cfg.speaker_mel.n_mels, embedding_dim=cfg.spk_dim)
        )

        self.text_encoder = TextEncoder(
            TextEncoderConfig(
                vocab_size=len(table), hidden=cfg.text_hidden, n_mels=cfg.audio.n_mels
            )
        )
        self.duration_predictor = DurationPredictor(cfg.text_hidden)
        self.acoustic = ConditionalFlowMatching(
            VectorFieldEstimator(
                EstimatorConfig(
                    n_mels=cfg.audio.n_mels, hidden=cfg.acoustic_hidden, spk_dim=cfg.spk_dim
                )
            )
        )
        self.vocoder: nn.Module = GriffinLimVocoder(cfg.audio)

    @torch.inference_mode()
    def embed_reference(self, reference: str | Path | np.ndarray | torch.Tensor) -> torch.Tensor:
        """从参考音频提取说话人 embedding，返回 (spk_dim,) 单位向量。"""
        if isinstance(reference, (str, Path)):
            wav, _ = load_wav(reference, sample_rate=self.config.speaker_mel.sample_rate)
        else:
            wav = np.asarray(reference, dtype=np.float32)
        wav = peak_normalize(wav)
        wav_t = torch.as_tensor(wav, dtype=torch.float32)
        mel = self.speaker_mel(wav_t).squeeze(0).transpose(0, 1)  # (T, mel_dim)
        return self.speaker_encoder.embed_utterance(mel)

    @torch.inference_mode()
    def synthesize_mel(
        self,
        text: str,
        speaker: torch.Tensor,
        language: str = "auto",
        duration_scale: float = 1.0,
    ) -> torch.Tensor:
        """给定文本与说话人 embedding，生成 log-mel，形状 (1, n_mels, T)。"""
        encoded = self.frontend.encode(text, language)
        ids = torch.tensor([encoded.ids], dtype=torch.long)
        mu_phone, hidden = self.text_encoder(ids)

        log_dur = self.duration_predictor(hidden)
        durations = torch.exp(log_dur) * duration_scale
        durations = durations.round().clamp(1, self.config.max_frames_per_token)
        mu_frame, _ = regulate_length(mu_phone, durations)

        speaker = speaker.unsqueeze(0) if speaker.dim() == 1 else speaker
        return self.acoustic.sample(mu_frame, speaker, n_timesteps=self.config.n_timesteps)

    @torch.inference_mode()
    def clone(
        self,
        text: str,
        reference: str | Path | np.ndarray | torch.Tensor,
        language: str = "auto",
    ) -> np.ndarray:
        """从参考音频克隆音色，合成 ``text`` 的波形（numpy float32）。"""
        speaker = self.embed_reference(reference)
        mel = self.synthesize_mel(text, speaker, language)
        wav = self.vocoder(mel)
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
