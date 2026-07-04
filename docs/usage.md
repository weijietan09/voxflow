# 使用指南

> ⚠️ 当前仓库处于研究早期阶段，随包分发的是**随机初始化**的模型，尚无预训练权重。
> 下面的示例主要用于演示接口与打通推理链路，输出音质不代表方法上限。

## 安装

```bash
pip install -e .            # 开发安装
pip install -e ".[dev]"     # 含测试 / lint 工具
```

依赖 PyTorch。CPU 环境建议：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Python API

```python
import numpy as np
from voxflow import VoiceCloner

cloner = VoiceCloner()

# 参考音频可以是 wav 路径，也可以是 numpy 波形
reference = "reference.wav"
wav = cloner.clone("你好，欢迎使用 VoxFlow。", reference, language="auto")

from voxflow.audio.io import save_wav
save_wav("out.wav", wav, cloner.config.audio.sample_rate)
```

只想要说话人 embedding：

```python
emb = cloner.embed_reference("reference.wav")   # (256,) 单位向量
```

只跑文本前端：

```python
from voxflow.text.frontend import TextFrontend

fe = TextFrontend()
print(fe.encode("语音克隆 voice cloning").tokens)
```

## 命令行

```bash
# 提取说话人 embedding 并存成 .npy
voxflow embed reference.wav -o speaker.npy

# 克隆音色合成文本
voxflow synth "你好世界" -r reference.wav -o out.wav --steps 10
```

## 配置

音频 / 特征参数由 `voxflow.config.AudioConfig` 描述，可从 YAML 读取：

```python
from voxflow.config import AudioConfig
cfg = AudioConfig.from_yaml("configs/base.yaml")
```

`configs/` 下另有说话人编码器与声学模型的参考配置。
