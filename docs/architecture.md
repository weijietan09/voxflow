# 架构

VoxFlow 把零样本声音克隆拆成四个相对独立的模块，彼此通过明确的张量接口衔接，
方便单独替换或训练。

```
参考音频 ──▶ 说话人编码器 ──▶ spk (256,)
                                     │
文本 ──▶ 文本前端 ──▶ ids ──▶ 文本编码器 ──▶ μ(逐音素) ──┐
                                     │                    ├─▶ 长度调节 ──▶ μ(逐帧)
                              时长预测器 ──▶ 时长 ────────┘        │
                                                                    ▼
                                              声学模型(CFM/扩散) ──▶ mel ──▶ 声码器 ──▶ 波形
```

## 1. 文本前端（`voxflow.text`）

- **归一化**：全角转半角、去零宽字符、中文数字读法。
- **分段**：按字符把文本切成中文 / 英文片段，标点并入相邻片段。
- **G2P**：中文用 `pypinyin` 得到声母 / 韵母 / 声调三类 token；英文用自带的
  小词典 + 逐字母回退得到 ARPAbet 音素。
- **符号表**：中英共享一套符号表，输出整数 id 序列。

## 2. 说话人编码器（`voxflow.models.speaker_encoder`）

GE2E 风格的 d-vector：3 层 LSTM 取末帧隐状态，线性投影后 ReLU 再 L2 归一化。
输入是 16 kHz / 40 维 log-mel。对整段音频采用 partial-utterance 平均，得到更稳定的
说话人 embedding。

## 3. 声学模型（`voxflow.models`）

核心是一个 1D U-Net 向量场估计器（`VectorFieldEstimator`），时间步经正弦嵌入 + MLP
注入每个残差块，中间层加一层自注意力。它被两种训练目标复用：

- **条件流匹配（CFM）**：最优传输路径 + 欧拉 ODE 采样，几步即可出结果；
- **DDPM 扩散**：噪声预测参数化，作为对照实现。

条件由文本编码器的逐帧均值 μ 与说话人 embedding 拼接提供。

## 4. 声码器（`voxflow.models.vocoder`）

- **HiFiGANLite**：转置卷积上采样 + 多感受野残差块，可训练；
- **GriffinLimVocoder**：免训练后备，用 mel 滤波器组伪逆 + Griffin-Lim 迭代
  重建相位，便于在没有声码器权重时先跑通链路。

## 数据与训练

`voxflow.data` 提供数据集与 collate；`voxflow.train` 提供损失函数与一个通用训练器。
完整的端到端训练还需要时长对齐（如 MAS），这部分仍在推进中，详见
[design-notes](./design-notes.md)。
