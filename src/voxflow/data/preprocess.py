"""数据预处理：把 manifest 里的音频预先算成 mel 存盘，便于反复训练。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from voxflow.audio.io import load_wav, peak_normalize
from voxflow.audio.mel import MelSpectrogram
from voxflow.config import AudioConfig
from voxflow.data.dataset import Sample
from voxflow.text.frontend import TextFrontend


def preprocess_manifest(
    samples: list[Sample],
    out_dir: str | Path,
    audio_config: AudioConfig | None = None,
    frontend: TextFrontend | None = None,
) -> list[dict[str, object]]:
    """把每条样本的 mel 存成 ``.npy``，并写出一个 ``index.json``。

    返回索引列表，每项含 mel 文件名、帧数与音素 id 序列。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = audio_config or AudioConfig()
    mel_extractor = MelSpectrogram(cfg)
    text_frontend = frontend or TextFrontend()

    index: list[dict[str, object]] = []
    for i, sample in enumerate(samples):
        wav, _ = load_wav(sample.audio_path, sample_rate=cfg.sample_rate)
        wav = peak_normalize(wav)
        mel = mel_extractor(torch.as_tensor(wav, dtype=torch.float32)).squeeze(0)
        mel_name = f"mel_{i:05d}.npy"
        np.save(out / mel_name, mel.numpy())
        index.append(
            {
                "mel": mel_name,
                "frames": int(mel.shape[1]),
                "ids": text_frontend.encode(sample.text, sample.language).ids,
            }
        )

    (out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index
