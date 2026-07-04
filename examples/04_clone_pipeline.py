"""示例 04：端到端声音克隆（随机初始化权重）。

用合成参考音频克隆音色，为一段中英混排文本合成波形并写成 wav。
注意：默认权重未经训练，输出接近噪声——这里演示的是完整推理链路是否打通。
"""

import numpy as np

from voxflow.audio.io import save_wav
from voxflow.pipeline import PipelineConfig, VoiceCloner


def main() -> None:
    rng = np.random.default_rng(42)
    reference = (0.1 * rng.standard_normal(16000)).astype(np.float32)

    config = PipelineConfig()
    config.n_timesteps = 8
    cloner = VoiceCloner(config)

    wav = cloner.clone("你好，这是 VoxFlow 的克隆示例。", reference)
    save_wav("clone_demo.wav", wav, config.audio.sample_rate)
    print(f"已写出 clone_demo.wav，共 {wav.size} 个采样点（{config.audio.sample_rate} Hz）")


if __name__ == "__main__":
    main()
