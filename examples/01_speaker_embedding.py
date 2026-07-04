"""示例 01：从参考音频提取说话人 embedding。

用一段合成的正弦波当作"参考音频"，跑通说话人编码器，打印 embedding 的
形状与模长（应为 1）。真实使用时把 `reference` 换成 wav 文件路径即可。
"""

import numpy as np

from voxflow.pipeline import VoiceCloner


def main() -> None:
    rng = np.random.default_rng(0)
    reference = (0.1 * rng.standard_normal(16000)).astype(np.float32)

    cloner = VoiceCloner()
    embedding = cloner.embed_reference(reference)

    print(f"embedding 维度: {embedding.shape[0]}")
    print(f"模长 (应≈1): {float(embedding.norm()):.4f}")


if __name__ == "__main__":
    main()
