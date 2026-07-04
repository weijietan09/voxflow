# API 参考

只列公开、较稳定的接口。研究早期阶段，签名仍可能变动。

## 顶层

```python
import voxflow
voxflow.__version__
```

顶层通过懒加载暴露：`VoiceCloner`、`PipelineConfig`、`TextFrontend`、
`AudioConfig`、`MelSpectrogram`、`SpeakerEncoder`、`ConditionalFlowMatching`、
`GaussianDiffusion`。

## `voxflow.pipeline`

### `class VoiceCloner(config=None, table=DEFAULT_TABLE)`

- `embed_reference(reference) -> Tensor`：从 wav 路径 / numpy 波形 / 张量提取
  说话人 embedding，形状 `(spk_dim,)`。
- `synthesize_mel(text, speaker, language="auto", duration_scale=1.0) -> Tensor`：
  生成 log-mel，形状 `(1, n_mels, T)`。
- `clone(text, reference, language="auto") -> np.ndarray`：端到端克隆，返回波形。

### `class PipelineConfig`

各子模块尺寸：`audio`、`speaker_mel`、`spk_dim`、`text_hidden`、`acoustic_hidden`、
`n_timesteps`、`max_frames_per_token`。

## `voxflow.text`

- `TextFrontend(table=DEFAULT_TABLE, add_bos_eos=True)`
  - `encode(text, language="auto") -> FrontendOutput`
  - `phonemize(text, language="auto") -> list[str]`
- `FrontendOutput`：字段 `text` / `tokens` / `ids`。
- `normalize(text, language="zh") -> str`、`number_to_chinese(n) -> str`

## `voxflow.audio`

- `MelSpectrogram(config) `：`forward(wav) -> (B, n_mels, T)`
- `load_wav(path, sample_rate=None) -> (wav, sr)`、`save_wav(path, wav, sr)`
- `resample(wav, orig_sr, target_sr)`、`peak_normalize(wav, peak=0.95)`

## `voxflow.models`

- `SpeakerEncoder(config)`：`forward(mels)`、`embed_utterance(mel)`
- `VectorFieldEstimator(config)`：`forward(x, t, mu, spk)`
- `ConditionalFlowMatching(estimator, sigma_min=1e-4)`：`compute_loss(...)`、`sample(...)`
- `GaussianDiffusion(estimator, n_steps=1000)`：`compute_loss(...)`、`sample(...)`
- `HiFiGANLite(...)`、`GriffinLimVocoder(config, n_iters=32)`

## `voxflow.train`

- `Trainer(model, conditioner, config=None, device="cpu")`：`train_step`、`fit`
- `TrainConfig`、`MultiResolutionSTFTLoss`、`masked_l1_loss`

## `voxflow.utils`

- `seed_everything(seed=1234, deterministic=False)`
- `get_logger(name="voxflow")`
- `save_checkpoint(...)`、`load_checkpoint(...)`
