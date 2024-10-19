"""命令行接口的单元测试。"""

import numpy as np
import pytest

from voxflow import cli


def test_version_action_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "voxflow" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_embed_writes_npy(tmp_path):
    from voxflow.audio.io import save_wav

    ref = tmp_path / "ref.wav"
    save_wav(ref, (0.1 * np.random.randn(16000)).astype(np.float32), 16000)
    out = tmp_path / "emb.npy"
    assert cli.main(["embed", str(ref), "-o", str(out)]) == 0
    assert out.exists()
    assert np.load(out).shape[0] == 256


def test_synth_writes_wav(tmp_path):
    from voxflow.audio.io import save_wav

    ref = tmp_path / "ref.wav"
    save_wav(ref, (0.1 * np.random.randn(16000)).astype(np.float32), 16000)
    out = tmp_path / "syn.wav"
    rc = cli.main(["synth", "你好", "-r", str(ref), "-o", str(out), "--steps", "2"])
    assert rc == 0
    assert out.exists()
