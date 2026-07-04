# VoxFlow

[![CI](https://github.com/weijietan09/voxflow/actions/workflows/ci.yml/badge.svg)](https://github.com/weijietan09/voxflow/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**VoxFlow** 是一个零样本声音克隆的 TTS 研究工具包：用少量参考音频克隆音色，
支持中英双语。它把「说话人编码器 + 流匹配 / 扩散声学模型 + 声码器」拆成清晰、
可单独替换的模块，强调可复现与离线优先。

> 🚧 **研究早期阶段。** 仓库当前提供的是完整的框架与推理链路，随包权重为随机
> 初始化——聚焦「架构 + 可训练 + 可跑通」，尚无预训练模型。接口仍可能变动。

## 特性

- 🗣️ **零样本克隆**：GE2E 风格 d-vector 说话人编码器，少量参考音频即可提取音色。
- 🌊 **流匹配声学模型**：条件流匹配（OT-CFM）+ 欧拉 ODE 采样，少步数出结果；
  另附 DDPM 扩散实现作对照。
- 🈶 **中英双语前端**：`pypinyin` 拼音 G2P + 自带英文 G2P，中英混排、代码切换。
- 🔊 **可插拔声码器**：HiFi-GAN-lite 生成器，或免训练的 Griffin-Lim 后备。
- 🧩 **模块化**：文本前端 / 说话人编码器 / 声学模型 / 声码器各自独立，接口明确。
- 🧪 **工程完善**：类型标注、单元测试、ruff + mypy + CI、CPU 可跑的小配置。

## 快速开始

```bash
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 环境
```

```python
from voxflow import VoiceCloner
from voxflow.audio.io import save_wav

cloner = VoiceCloner()
wav = cloner.clone("你好，欢迎使用 VoxFlow。", "reference.wav", language="auto")
save_wav("out.wav", wav, cloner.config.audio.sample_rate)
```

命令行：

```bash
voxflow embed reference.wav -o speaker.npy
voxflow synth "你好世界" -r reference.wav -o out.wav --steps 10
```

更多见 [`examples/`](./examples) 与 [使用指南](./docs/usage.md)。

## 架构

```
参考音频 ─▶ 说话人编码器 ─▶ spk
文本 ─▶ 文本前端 ─▶ 文本编码器/时长 ─▶ μ ─▶ 声学模型(CFM/扩散) ─▶ mel ─▶ 声码器 ─▶ 波形
```

详见 [架构文档](./docs/architecture.md) 与 [设计笔记](./docs/design-notes.md)。

## 文档

- [使用指南](./docs/usage.md)
- [架构](./docs/architecture.md)
- [设计笔记](./docs/design-notes.md)
- [API 参考](./docs/api-reference.md)

## 开发

```bash
ruff check . && ruff format --check .
mypy src/voxflow
pytest --cov=voxflow
```

欢迎参与，见 [贡献指南](./CONTRIBUTING.md)。

## 路线图

- [ ] 时长对齐（单调对齐搜索 MAS），打通端到端训练
- [ ] 在公开中英数据集上训练并发布权重
- [ ] 更强的英文 G2P 后端
- [ ] 声码器对抗训练与流式推理

## 许可证

[MIT](./LICENSE)
