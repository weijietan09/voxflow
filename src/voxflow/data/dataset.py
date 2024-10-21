"""文本-mel 数据集与 collate。

manifest 为纯文本，每行 ``音频路径|文本|语言``（语言可省略，默认 auto）。
数据集在取样时即时算 mel，collate 负责把变长的 id 序列与 mel 右侧零填充。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from voxflow.audio.io import load_wav, peak_normalize
from voxflow.audio.mel import MelSpectrogram
from voxflow.config import AudioConfig
from voxflow.text.frontend import TextFrontend
from voxflow.text.symbols import DEFAULT_TABLE, SymbolTable


@dataclass
class Sample:
    """一条训练样本。"""

    audio_path: str
    text: str
    language: str = "auto"


def read_manifest(path: str | Path) -> list[Sample]:
    """解析 manifest 文件，返回 :class:`Sample` 列表。"""
    samples: list[Sample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            raise ValueError(f"manifest 行格式错误（至少需要 路径|文本）: {line!r}")
        language = parts[2] if len(parts) > 2 else "auto"
        samples.append(Sample(parts[0], parts[1], language))
    return samples


class TextMelDataset(Dataset):
    """从 :class:`Sample` 列表按需产出 (ids, mel)。"""

    def __init__(
        self,
        samples: list[Sample],
        audio_config: AudioConfig | None = None,
        table: SymbolTable = DEFAULT_TABLE,
        frontend: TextFrontend | None = None,
    ) -> None:
        self.samples = list(samples)
        self.audio_config = audio_config or AudioConfig()
        self.mel = MelSpectrogram(self.audio_config)
        self.frontend = frontend or TextFrontend(table)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        ids = torch.tensor(self.frontend.encode(sample.text, sample.language).ids, dtype=torch.long)
        wav, _ = load_wav(sample.audio_path, sample_rate=self.audio_config.sample_rate)
        wav = peak_normalize(wav)
        mel = self.mel(torch.as_tensor(wav, dtype=torch.float32)).squeeze(0)
        return {"ids": ids, "mel": mel}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """把一批 (ids, mel) 右侧零填充成规整张量，并附带各自长度。"""
    ids = [item["ids"] for item in batch]
    mels = [item["mel"] for item in batch]
    id_lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    mel_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)

    n_mels = mels[0].shape[0]
    ids_padded = torch.zeros(len(batch), int(id_lengths.max()), dtype=torch.long)
    mel_padded = torch.zeros(len(batch), n_mels, int(mel_lengths.max()))
    for i, (seq, mel) in enumerate(zip(ids, mels, strict=True)):
        ids_padded[i, : seq.shape[0]] = seq
        mel_padded[i, :, : mel.shape[1]] = mel

    return {
        "ids": ids_padded,
        "id_lengths": id_lengths,
        "mel": mel_padded,
        "mel_lengths": mel_lengths,
    }
