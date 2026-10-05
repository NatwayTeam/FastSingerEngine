![FastSinger](https://www.ttfont.com/preview/1/12723130/000000/95/FastSinger "FastSingerLogo")

---

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-FastSingerEngine-181717?logo=github)](https://github.com/NatwayTeam/FastSingerEngine)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

> 以深度神经网络为基础构建的新一代歌声合成引擎！

> [!NOTE]
>目前项目处于早期开发阶段，不代表最终正式版效果。
>本项目是为虚拟歌姬打造的合成引擎，一切与虚拟歌姬无关的声库，皆与本项目的愿景背道而驰，我们既不会帮助其宣传，更不会收录其声库。

## 总体架构

**推理链路** ：拼音 和音素序列 · `Encoder` · `VarianceAdaptor` · 80 维梅尔频谱 · `PostNet` 残差 · HiFi-GAN · 22050 Hz 波形

**训练链路** ：波形 · 梅尔频谱 、 F0 、 能量 和 时长自动提取 · 生成器 · L1 + 能量 MSE + 三路对抗损失 · `best` / `last` checkpoint

**设计特色** ：非自回归 · 无时长预测器 · 无音高预测器 · 声码器冻结 · 音素切分零人工标注 · 对抗项延迟开启并动态调权 · 多重声学模型对抗训练

> [!NOTE]
>生成器不含时长预测器与音高预测器，时长 与 音高 必须由调用方提供：训练阶段来自预处理（RMVPE 提取 F0、梅尔频谱自动切分时长），推理阶段由节拍参数推导。模型因此不带音高条件，音高完全由外部 F0 曲线决定。

## 文件目录

```
FastSinger/
├── Maker.py             预处理 · 训练 · 验证
├── Loader.py            推理
├── config/
│   ├── model.yaml       模型结构、音频与频谱参数
│   ├── train.yaml       路径、优化器、步数调度、对抗训练
│   └── infer.yaml       推理模型目录与声码器位置
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── data/                训练时所需的WAV音频（不入库）
├── preprocessed/        预处理产物（不计入仓库）
├── models/              所有模型权重（不计入仓库）
│   ├── rmvpe.pt
│   └── hifigan/
│       ├── config.json
│       └── HifiGan.pth
└── out/                 TensorBoard 日志与推理输出（不计入仓库）
```

`Maker.py` 与 `Loader.py` 为训练与推理全部实现，合计约四千行，其余均为配置与文档。

## 使用方法

### 一、安装依赖

```bash
pip install -r requirements.txt
```

依赖版本已在 `requirements.txt` 固定，测试时环境为 Python 3.11。

### 二、准备权重

Rmvpe和HifiGan的权重不在仓库中，缺失时程序将直接报错退出，须先行放置到位。

`rmvpe.pt` 是训练时预处理提取音高所需；`hifigan/` 下两个文件训练与推理均需。路径在 `config/train.yaml` 与 `config/infer.yaml` 的 `vocoder` 段配置。

### 三、准备数据

将 WAV 置于 `./data`（`config/train.yaml` 的 `path.corpus_path`），文件名须匹配：文件名(拼音)(MIDI 音号)(时长)。

采样率无需自行处理，会重采样至 22050 Hz 并混合为单声道。

音素切分完全自动，不依赖任何人工标注。分界点在梅尔频谱上自动检测，整个预处理仅读取 WAV 文件及其文件名，不读取转写或对齐文件。

### 四、训练

```bash
python Maker.py train
```

该命令依次完成预处理与训练，无独立预处理子命令。启动后依次输出学习率调度、对抗训练参数与参数量，随后进入进度条。

> [!NOTE]
> **断点续训为自动行为。** 启动时读取`last`，存在则自其步数继续，不存在则从零开始，无需额外参数。`best` 按验证损失保存且仅含生成器不含判别器，其指标可与非对抗训练直接比较。

主要配置位于 `config/train.yaml`：`step.total_step` 为总步数（默认 10000），`optimizer.lr` 为学习率峰值（0.0002），`min_lr` 为余弦衰减下限，`warmup_step` 为线性预热步数。对抗训练的开关、起始步数与各项权重位于 `adversarial` 段。各项含义、取值范围与填法见「配置说明」。

### 五、推理

```bash
python Loader.py '{"pinyin":"zhuang","bpm":120,"bars":1,"midi":60,"curve":["+0.0","+0.1"]}'
```

配置采用相对路径，须在仓库根目录下执行。输出写入 `out/{拼音}_{音高}.wav`，例如 `out/zhuang_60.wav`，采样率 22050 Hz，写入前峰值归一化至 0.92。

输入 JSON 字段：

- `pinyin`（必填）——拼音音节，自动转小写。
- `bpm`（必填）——速度，须大于 0。
- `bars`（必填）——小节数，须大于 0。
- `midi`（必填）——基准 MIDI 音号，69 对应 A4 = 440 Hz。
- `num`（可选，默认 4）——拍号分子。
- `den`（可选，默认 4）——拍号分母。
- `curve`（可选，默认 `["+0.0"]`）——半音偏移序列，线性插值后叠加至 `midi`，用于滑音。

推理阶段无时长预测器，时长与音高均由上述参数推导：总帧数按 `bars × num × (60/bpm) × (4/den)` 换算；音素分配依据 `stats.json` 记录的训练期平均时长，首音素取其均值、余量归尾音素，结果确定。F0 由 `midi` 与 `curve` 插值换算，清音声母对应帧置零。

> [!NOTE]
> `pinyin` 必须出现在训练语料中，否则程序直接退出——推理依赖预处理生成的 `preprocessed/pinyin2phones.json`，语料外拼音无兜底推导。此外，训练期 F0 来自 RMVPE 对真实音频的提取，含自然颤音与噪声，推理期 F0 为理想平滑曲线，两者分布不一致，将影响最终听感。

推理仅需 `models/FastSinger.pth`、`models/hifigan/`、`preprocessed/stats.json` 与 `preprocessed/pinyin2phones.json`，不需要 `rmvpe.pt` 及各 `.npy` 中间文件。

## 配置说明

三份配置各有分工，改动前先确认改对了文件：

| 文件 | 谁读取 | 管什么 |
| --- | --- | --- |
| `config/model.yaml` | `Maker.py` 与 `Loader.py` 都读 | 模型结构、音频与频谱参数、预处理产物目录 |
| `config/train.yaml` | 仅 `Maker.py` | 语料与权重路径、优化器、步数调度、对抗训练 |
| `config/infer.yaml` | 仅 `Loader.py` | 生成器权重与声码器位置、推理期能量控制 |

> [!NOTE]
> `Maker.py` 启动时把 `train.yaml` 合并进 `model.yaml`（同名键以 `train.yaml` 为准），而 `Loader.py` 只读 `model.yaml`。因此凡是训练与推理都要用的键（`path.preprocessed_path`、`audio.*`、`transformer.*` 等）必须写在 `model.yaml`，写进 `train.yaml` 会导致推理读不到。

### `config/model.yaml`

| 配置项 | 默认值 | 含义与填法 |
| --- | --- | --- |
| `path.preprocessed_path` | `./preprocessed` | 预处理产物目录。训练时写入，训练与推理都从这里读 `stats.json`、`pinyin2phones.json` 与各 `.npy`；改后需重新执行 `python Maker.py train`。 |
| `audio.sampling_rate` | `22050` | 输出采样率，须与 HiFi-GAN 的 `config.json` 一致。改动等于换整套音频前端，必须重新预处理并重训。 |
| `audio.hop_length` | `256` | 每帧样本数，帧率 = `22050 / 256` ≈ 86 帧/秒。时长、F0、能量都按它对齐，改动须重新预处理。 |
| `audio.max_wav_value` | `32768.0` | 声码器输出由 `[-1, 1]` 还原成 `int16` 的系数，固定 `32768.0`，不要改。 |
| `audio.n_mel_channels` | `80` | 梅尔频谱维数，固定 80，与已有权重绑定。 |
| `transformer.encoder_layer` | `4` | 编码器层数。 |
| `transformer.encoder_head` | `2` | 注意力头数，须整除 `encoder_hidden`。 |
| `transformer.encoder_hidden` | `256` | 隐层维度，同时是音素 embedding 与能量 embedding 的宽度，改动等于换模型。 |
| `transformer.conv_filter_size` | `1024` | 前馈层中间维数，惯例取 `4 × encoder_hidden`。 |
| `transformer.conv_kernel_size` | `[9, 1]` | 前馈两层卷积的核大小，一般不动。 |
| `transformer.encoder_dropout` | `0.2` | 编码器 dropout。过拟合时调高，欠拟合时可降到 `0.1`。 |
| `variance_predictor.filter_size` | `256` | 能量预测器的卷积通道数。时长与音高不由模型预测，该段只服务能量。 |
| `variance_predictor.kernel_size` | `3` | 能量预测器的卷积核大小。 |
| `variance_predictor.dropout` | `0.5` | 能量预测器的 dropout。 |
| `variance_embedding.pitch_quantization` | `log` | F0 分桶方式，只接受 `log` 或 `linear`。`log` 更贴近音高感知，建议保持。 |
| `variance_embedding.energy_quantization` | `linear` | 能量分桶方式，同样只接受 `linear` 或 `log`。 |
| `variance_embedding.n_bins` | `256` | 分桶数量，决定两张 embedding 表的大小，须与已有权重一致。 |
| `max_seq_len` | `1000` | 位置编码表长度。训练期序列超过它会因形状不匹配报错，推理期超长则改用即时正弦编码。单音节只有 1~2 个音素，保持默认即可。 |

### `config/train.yaml`

**路径与权重**

| 配置项 | 默认值 | 含义与填法 |
| --- | --- | --- |
| `path.corpus_path` | `./data` | 训练 WAV 所在目录，只扫描该目录第一层的 `.wav`，不递归子目录。文件名格式见「三、准备数据」。 |
| `path.ckpt_path` | `./models` | 权重写出目录，同时是 `rmvpe.pt` 的查找位置，缺文件直接报错退出。 |
| `path.log_path` | `./out/log` | TensorBoard 日志根目录，训练与验证分别写入其 `train/`、`val/` 子目录。 |
| `path.result_path` | `./out` | 启动时创建该目录，此外不被读取：日志由 `path.log_path` 决定，推理输出写死在 `out/`，属保留字段。 |
| `checkpoint.best_file` | `FastSinger.pth` | 验证最优权重的文件名，必填，缺失直接退出。续训用的 `last` 由它派生为 `FastSinger(last).pth`，两者同目录。 |
| `vocoder.path` | `./models/hifigan` | HiFi-GAN 目录，必填。 |
| `vocoder.file` | `HifiGan.pth` | 声码器权重文件名，必填。 |
| `vocoder.config` | `config.json` | 声码器结构配置文件名，缺省即 `config.json`。 |

**预处理**

| 配置项 | 默认值 | 含义与填法 |
| --- | --- | --- |
| `preprocessing.val_size` | `1000` | 固定随机种子打乱后取前 N 条作验证集，其余作训练集。**语料少于该值时训练集为空**，启动即断言失败，须满足「语料条数 − `val_size`」> `4 × batch_size`。 |
| `preprocessing.text.language` | `zh` | 保留字段，当前代码不读取，改不改都不影响结果。 |
| `preprocessing.stft.filter_length` | `1024` | FFT 点数。 |
| `preprocessing.stft.win_length` | `1024` | 窗长，不得超过 `filter_length`。 |
| `preprocessing.mel.mel_fmin` / `mel_fmax` | `0` / `8000` | 梅尔滤波器组的频率下限与上限（Hz），`mel_fmax` 不得超过采样率的一半（11025）。 |
| `preprocessing.energy.normalization` | `True` | 能量是否按均值方差标准化，改后 `stats.json` 随之变化。 |

每次 `python Maker.py train` 都会无条件重跑一遍预处理，上述特征相关配置改完直接重新执行该命令即可，无需手动清理 `preprocessed/`。

**优化器与步数**

| 配置项 | 默认值 | 含义与填法 |
| --- | --- | --- |
| `optimizer.lr` | `0.0002` | 学习率峰值，warmup 结束时达到。 |
| `optimizer.min_lr` | `0.00002` | 余弦衰减的下限。 |
| `optimizer.warmup_step` | `500` | 线性 warmup 的步数，填 `0` 表示不 warmup。 |
| `optimizer.batch_size` | `16` | 单次前向的样本数。 |
| `optimizer.grad_acc_step` | `1` | 累积多少次前向才更新一次参数，有效批大小 = `batch_size × grad_acc_step`，显存不够时调大它。 |
| `optimizer.grad_clip_thresh` | `5.0` | 梯度范数裁剪阈值。 |
| `optimizer.betas` / `eps` / `weight_decay` | `[0.9, 0.98]` / `1e-9` / `0.0` | Adam 的标准参数，一般不动。 |
| `step.total_step` | `10000` | 总步数，既是训练终点，也是余弦衰减的跨度。 |
| `step.log_step` | `100` | 每 N 步打印一次损失并写入 TensorBoard 标量。 |
| `step.synth_step` | `500` | 每 N 步把一条训练样本的重建与合成音频、梅尔图写入 TensorBoard。 |
| `step.val_step` | `500` | 每 N 步跑一次验证，`mel` + `PostNet` 两路 L1 之和创新低时保存 `best`。 |
| `step.save_step` | `1000` | 每 N 步保存一次 `last`，断点续训读的就是它。 |

步数以「单次前向」为单位计数，`grad_acc_step > 1` 时一次参数更新会跨多个 step。

**对抗训练**

三个判别器都作用在生成的梅尔频谱上（声码器始终冻结），更细的注释见 `config/train.yaml` 内联说明。

| 配置项 | 默认值 | 含义与填法 |
| --- | --- | --- |
| `adversarial.enabled` | `True` | 对抗训练总开关，`False` 退化为纯 L1 + 能量 MSE。 |
| `adversarial.start_step` | `3000` | 从第几步开始引入对抗项，太早开对抗会毁掉重建。 |
| `adversarial.ramp_steps` | `1000` | 对抗权重由 0 线性升到目标值所用的步数。 |
| `adversarial.adv_weight` | `0.05` | 对抗（hinge）损失的基础权重。 |
| `adversarial.fm_weight` | `0.2` | feature matching 损失的基础权重。 |
| `adversarial.phone_weight` | `0.1` | 逐帧音素交叉熵的权重，属监督锚点，不参与动态缩放。 |
| `adversarial.pitch_weight` | `0.05` | （梅尔频谱, F0）判别损失的权重。 |
| `adversarial.d_lr` / `d_min_lr` | `0.0001` / `0.00001` | 判别器学习率峰值与余弦下限，warmup 与 `total_step` 沿用生成器的设置。 |
| `adversarial.d_iters` | `1` | 每个 step 判别器更新次数。 |
| `adversarial.base_channels` | `32` | 判别器基础通道数。 |
| `adversarial.auto_gain` | `True` | 动态调权开关，按判别器强弱与高频能量两个信号闭环调节对抗项权重。 |
| `adversarial.gain_min` / `gain_max` | `0.01` / `0.15` | 对抗项实际权重的下限与上限（等效 `adv_weight` 的取值范围），关闭 `auto_gain` 后不生效。 |
| `adversarial.eq_target` | `0.6` | 期望的判别器强度，达到该强度即不再加力。 |
| `adversarial.ctrl_k` | `0.05` | 调节的比例系数，越大收放越快。 |
| `adversarial.ctrl_every` | `25` | 每 N 步调节一次权重。 |
| `adversarial.hi_bands` | `12` | 统计高频能量差时只看顶部多少条梅尔频带。 |
| `adversarial.hi_ceiling` | `0.2` | 高频能量差超过该值即强制降权，用于抑制嘶声。 |
| `adversarial.cls_train_until` | `4000` | 音素分类器训练到该步后冻结，使音素项成为静止锚点；留空则自动取 `start_step + ramp_steps`。 |
| `adversarial.cls_ce_weight` | `0.1` | 分类器在判别器目标中的权重，冻结后失效。 |

### `config/infer.yaml`

| 配置项 | 默认值 | 含义与填法 |
| --- | --- | --- |
| `path.model_path` | `./models` | 生成器权重所在目录。 |
| `checkpoint.infer_file` | `FastSinger.pth` | 要加载的权重文件名，必填，缺失直接退出。通常填验证最优的 `best`；`FastSinger(last).pth` 同样含生成器权重，可加载，但另带判别器且不是最优权重。 |
| `vocoder.path` | `./models/hifigan` | HiFi-GAN 目录，必填。 |
| `vocoder.file` | `HifiGan.pth` | 声码器权重文件名，必填。 |
| `vocoder.config` | `config.json` | 声码器结构配置文件名。 |
| `energy_control` | `1.0` | 能量乘性系数，即模型预测能量 × 该值。`> 1` 更响、起伏更大，`< 1` 更轻更平，`1.0` 按预测原样。只在推理生效，训练始终使用真实能量。 |

推理还要读 `model.yaml` 的 `path.preprocessed_path`、`audio.*` 与 `transformer.*` 等，必须与训练时保持一致：结构参数对不上会加载失败，特征参数对不上则听感异常。

> [!TIP]
> - 只改 `optimizer.*`、`step.*`、`adversarial.*`、`energy_control`：不影响特征与模型结构，重新执行 `python Maker.py train` 即可续训，或直接重新推理。
> - 改 `audio.*`、`preprocessing.*`：特征分布变了，删掉 `models/` 下的旧权重后重跑，否则续训会接着旧权重吃新特征。
> - 改 `transformer.*`、`variance_*`、`max_seq_len`：模型结构变了，旧权重加载会直接失败，删掉旧权重重训。

## 致谢

感谢 **[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)**，该仓库为本项目早期开发提供了核心思路，`Encoder`、`VarianceAdaptor`、`PostNet`、`LengthRegulator` 等结构的设计与实现均参考自它。本仓库 `LICENSE` 文件亦沿用其版权声明（Chung-Ming Chien，MIT）。

同时感谢 [xcmyz/FastSpeech](https://github.com/xcmyz/FastSpeech)（ming024 实现的上游基础）、[jik876/hifi-gan](https://github.com/jik876/hifi-gan)（声码器，见 `models/LICENSE`）、[FastSpeech 2](https://arxiv.org/abs/2006.04558v1) 论文，以及 [RMVPE](https://arxiv.org/abs/2306.15412) 的 F0 提取方法。

