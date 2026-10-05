![FastSinger](https://www.ttfont.com/preview/1/12723130/000000/95/FastSinger "FastSingerLogo")

---

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-FastSingerEngine-181717?logo=github)](https://github.com/NatwayTeam/FastSingerEngine)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

> 以深度神经网络为基础构建的新一代歌声合成引擎！

> [!NOTE]
>目前项目处于早期开发阶段，不代表最终正式版效果。
>本项目是为虚拟歌姬打造的合成引擎，一切与虚拟歌姬无关的声库，皆与本项目的愿景背道而驰，我们既不会帮助其宣传，更不会收录其声库。

---

## 总体架构

**推理链路** ：拼音 和音素序列 · `Encoder` · `VarianceAdaptor` · 80 维梅尔频谱 · `PostNet` 残差 · HiFi-GAN · 22050 Hz 波形

**训练链路** ：波形 · 梅尔频谱 、 F0 、 能量 和 时长自动提取 · 生成器 · L1 + 能量 MSE + 三路对抗损失 · `best` / `last` checkpoint

**设计特色** ：非自回归 · 无时长预测器 · 无音高预测器 · 声码器冻结 · 音素切分零人工标注 · 对抗项延迟开启并动态调权 · 多重声学模型对抗训练

> [!NOTE]
>生成器不含时长预测器与音高预测器，时长 与 音高 必须由调用方提供：训练阶段来自预处理（RMVPE 提取 F0、梅尔频谱自动切分时长），推理阶段由节拍参数推导。模型因此不带音高条件，音高完全由外部 F0 曲线决定。

---

## 文件目录

```
FastSinger/
├── Maker.py            预处理 · 训练 · 验证
├── Loader.py           推理
├── config/
│   ├── model.yaml   模型结构、音频与频谱参数
│   ├── train.yaml      路径、优化器、步数调度、对抗训练
│   └── infer.yaml      推理模型目录与声码器位置
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── data/                  训练时所需的WAV音频（不入库）
├── preprocessed/     预处理产物（不计入仓库）
├── models/              所有模型权重（不计入仓库）
│   ├── rmvpe.pt
│   └── hifigan/
│       ├── config.json
│       └── HifiGan.pth
└── out/                    TensorBoard 日志与推理输出（不计入仓库）
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

主要配置位于 `config/train.yaml`：`step.total_step` 为总步数（默认 10000），`optimizer.lr` 为学习率峰值（0.0002），`min_lr` 为余弦衰减下限，`warmup_step` 为线性预热步数。对抗训练的开关、起始步数与各项权重位于 `adversarial` 段。

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

## 致谢

感谢 **[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)**，该仓库为本项目早期开发提供了核心思路，`Encoder`、`VarianceAdaptor`、`PostNet`、`LengthRegulator` 等结构的设计与实现均参考自它。本仓库 `LICENSE` 文件亦沿用其版权声明（Chung-Ming Chien，MIT）。

同时感谢 [xcmyz/FastSpeech](https://github.com/xcmyz/FastSpeech)（ming024 实现的上游基础）、[jik876/hifi-gan](https://github.com/jik876/hifi-gan)（声码器，见 `models/LICENSE`）、[FastSpeech 2](https://arxiv.org/abs/2006.04558v1) 论文，以及 [RMVPE](https://arxiv.org/abs/2306.15412) 的 F0 提取方法。

