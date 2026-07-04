# 示例

这些脚本用合成数据打通各个模块，不依赖外部数据集，可直接运行：

```bash
python examples/01_speaker_embedding.py   # 提取说话人 embedding
python examples/02_text_frontend.py       # 中英文本前端
python examples/03_flow_matching_toy.py   # 玩具流匹配训练 + 采样
python examples/04_clone_pipeline.py      # 端到端克隆（随机权重）
```

> 默认权重是随机初始化的，示例 04 的输出音频接近噪声。示例的目的是验证
> 数据流与接口，而非音质；训练好的权重需要自行准备。
