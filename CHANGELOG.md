# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)，
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### Added
- 端到端训练所需的时长对齐（计划中）。

## [0.1.0] - 研究早期基础版本

### Added
- 中英双语文本前端：归一化、数字读法、拼音 / 英文 G2P、共享符号表。
- GE2E 风格 LSTM 说话人编码器，支持 partial-utterance 平均。
- 1D U-Net 向量场估计器，条件流匹配（OT-CFM）与 DDPM 扩散两种声学模型。
- HiFi-GAN-lite 生成器与 Griffin-Lim 免训练声码器。
- 高层 `VoiceCloner` 流水线与 `voxflow` 命令行。
- 数据集 / collate / 预处理、通用训练器与 STFT / mel 损失。
- 完整工程设施：ruff、mypy、pytest、GitHub Actions CI、文档与示例。

### Known limitations
- 随包权重为随机初始化，尚无预训练模型。
- 未实现时长对齐，端到端训练尚不完整。

[Unreleased]: https://github.com/weijietan09/voxflow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/weijietan09/voxflow/releases/tag/v0.1.0
