"""数据集与 collate 的单元测试。"""

import torch

from voxflow.data.dataset import Sample, TextMelDataset, collate, read_manifest


def test_read_manifest(tmp_path):
    manifest = tmp_path / "train.txt"
    manifest.write_text("a.wav|你好\n# 注释\nb.wav|hello|en\n", encoding="utf-8")
    samples = read_manifest(manifest)
    assert len(samples) == 2
    assert samples[0] == Sample("a.wav", "你好", "auto")
    assert samples[1].language == "en"


def test_collate_pads_ids_and_mels():
    batch = [
        {"ids": torch.tensor([1, 2, 3]), "mel": torch.randn(8, 10)},
        {"ids": torch.tensor([4, 5]), "mel": torch.randn(8, 6)},
    ]
    out = collate(batch)
    assert out["ids"].shape == (2, 3)
    assert out["mel"].shape == (2, 8, 10)
    assert out["id_lengths"].tolist() == [3, 2]
    assert out["mel_lengths"].tolist() == [10, 6]
    # 第二条被填充的 id 应为 0
    assert out["ids"][1, 2].item() == 0


def test_dataset_getitem(make_wav):
    wav = make_wav(seconds=0.5)
    dataset = TextMelDataset([Sample(str(wav), "你好world")])
    assert len(dataset) == 1
    item = dataset[0]
    assert item["ids"].dtype == torch.long
    assert item["mel"].shape[0] == 80
    assert item["mel"].shape[1] > 0
