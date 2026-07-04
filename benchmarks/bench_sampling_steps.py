"""基准：条件流匹配的采样步数对推理速度（RTF）的影响。

RTF（real-time factor）= 合成耗时 / 音频时长，越小越好。
用随机初始化的小模型跑，纯粹衡量不同步数下的速度量级。
"""

import time

import torch

from voxflow.config import AudioConfig
from voxflow.models.flow_matching import ConditionalFlowMatching
from voxflow.models.velocity_field import EstimatorConfig, VectorFieldEstimator


def main() -> None:
    torch.manual_seed(0)
    cfg = AudioConfig()
    model = ConditionalFlowMatching(
        VectorFieldEstimator(EstimatorConfig(n_mels=cfg.n_mels, hidden=64))
    )
    model.eval()

    length = 200  # mel 帧数
    audio_seconds = length * cfg.hop_length / cfg.sample_rate
    mu = torch.randn(1, cfg.n_mels, length)
    spk = torch.randn(1, 256)

    print(f"音频时长约 {audio_seconds:.2f}s，逐步测速：")
    for steps in [1, 2, 4, 8, 16, 32]:
        model.sample(mu, spk, n_timesteps=steps)  # warmup
        start = time.perf_counter()
        for _ in range(3):
            model.sample(mu, spk, n_timesteps=steps)
        elapsed = (time.perf_counter() - start) / 3
        print(f"  steps={steps:3d}  {elapsed * 1000:7.1f} ms  RTF={elapsed / audio_seconds:.3f}")


if __name__ == "__main__":
    main()
