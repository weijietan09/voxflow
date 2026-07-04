"""VoxFlow 命令行入口。

子命令：
- ``embed``：从参考音频提取说话人 embedding；
- ``synth``：用参考音频克隆音色，合成给定文本（在后续提交里加入）。

重量级依赖（torch、流水线）在各子命令内部按需导入，
这样 ``voxflow --version`` / ``--help`` 无需加载 torch。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from voxflow.version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxflow", description="零样本声音克隆 TTS 工具包")
    parser.add_argument("--version", action="version", version=f"voxflow {__version__}")
    sub = parser.add_subparsers(dest="command")

    embed = sub.add_parser("embed", help="从参考音频提取说话人 embedding")
    embed.add_argument("reference", type=Path, help="参考音频文件（wav/flac 等）")
    embed.add_argument("-o", "--output", type=Path, default=None, help="保存为 .npy 文件")

    synth = sub.add_parser("synth", help="用参考音频克隆音色，合成给定文本")
    synth.add_argument("text", help="要合成的文本（支持中英混排）")
    synth.add_argument("-r", "--reference", type=Path, required=True, help="参考音频文件")
    synth.add_argument("-o", "--output", type=Path, default=Path("output.wav"), help="输出 wav 路径")
    synth.add_argument("-l", "--language", default="auto", help="语言：auto/zh/en")
    synth.add_argument("--steps", type=int, default=10, help="流匹配采样步数")

    return parser


def _cmd_embed(args: argparse.Namespace) -> int:
    import numpy as np

    from voxflow.pipeline import VoiceCloner

    cloner = VoiceCloner()
    embedding = cloner.embed_reference(args.reference).cpu().numpy()
    if args.output is not None:
        np.save(args.output, embedding)
        print(f"已保存 embedding 到 {args.output}（维度 {embedding.shape[0]}）")
    else:
        print(f"embedding 维度={embedding.shape[0]} 模长={float(np.linalg.norm(embedding)):.4f}")
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from voxflow.audio.io import save_wav
    from voxflow.pipeline import PipelineConfig, VoiceCloner

    config = PipelineConfig()
    config.n_timesteps = args.steps
    cloner = VoiceCloner(config)
    wav = cloner.clone(args.text, args.reference, args.language)
    save_wav(args.output, wav, config.audio.sample_rate)
    print(f"已写出 {args.output}（{wav.size} 采样点，{config.audio.sample_rate} Hz）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "embed":
        return _cmd_embed(args)
    if args.command == "synth":
        return _cmd_synth(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
